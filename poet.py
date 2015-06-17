import numpy
from numpy import array

import theano
from theano import tensor

from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.bricks import Tanh, Initializable
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import LSTM, RecurrentStack
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)

from blocks.config import config
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.saveload import SimpleExtension
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation

from fuel.datasets import OneBillionWord, TextFile
from fuel.transformers import Mapping, Batch, Padding, Filter
from fuel.schemes import ConstantScheme

config.recursion_limit = 100000
floatX = theano.config.floatX

# Dictionaries
all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', '!', '?', '<UNK>'] +
             [' ', '<S>', '</S>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}

def _lower(s):
    return s.lower()

def _transpose(data):
    return tuple(array.T for array in data)

def _filter_long(data):
    return len(data[0]) <= 100

class Poet(Initializable):
    """Poet is your favourite text-generation artist.
    Poet should learn your favourite topics, and talk to you about them.
    """
    def __init__(self, dimension, depth, alphabet_size, **kwargs):
        super(Poet, self).__init__(**kwargs)

        lookup = LookupTable(alphabet_size, dimension)

        recurrents = [LSTM(dim=dimension) for _ in range(depth)]

        transition = RecurrentStack( recurrents,
            name="transition", skip_connections = True)

        readout = Readout(
            readout_dim=alphabet_size,
            source_names=[transition.apply.states[0]],
            emitter=SoftmaxEmitter(name="emitter"),
            feedback_brick=LookupFeedback(alphabet_size, dimension),
            name="readout")
        
        generator = SequenceGenerator(
            readout=readout,
            transition=transition,
            name="generator")

        self.lookup = lookup
        self.generator = generator
        self.children = [lookup, generator]

    @application
    def cost(self, chars, chars_mask):
        return aggregation.mean(self.generator.cost_matrix(
            chars, chars_mask).sum(), chars.shape[1])

class Write(SimpleExtension):
    """Make your poet write
    Parameters
    ----------
    chars : int Number of characters to generate
    """
    def __init__(self, poet, chars=100, **kwargs):
        super(Write, self).__init__(**kwargs)
        steps = 300
        sample = ComputationGraph(poet.generator.generate(n_steps=steps, 
            batch_size=1, iterate=True))
        self.sample_fn = sample.get_theano_function()

    def do(self, callback_name, *args):
        _1, _2, _3, _4, _5, _6, outputs, _8 = self.sample_fn()

        try:
            true_length = list(outputs).index(array(char2code['</S>'])) + 1
        except ValueError:
            true_length = len(outputs)
        outputs = outputs[:true_length]

        print "".join(code2char[code[0]] for code in outputs)


poet = Poet(dimension = 200, depth = 3, alphabet_size = len(char2code), name="poet")
poet.weights_init = IsotropicGaussian(0.1)
poet.biases_init = Constant(0.0)

dataset_options = dict(dictionary=char2code, level="character",
                               preprocess=_lower)

#dataset = TextFile(["poet.py"], **dataset_options)
dataset = OneBillionWord("training", [99], **dataset_options)

data_stream = dataset.get_example_stream()
data_stream = Filter(data_stream, _filter_long)
data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10))
data_stream = Padding(data_stream)
data_stream = Mapping(data_stream, _transpose)

# Build the cost computation graph
chars = tensor.lmatrix("features")
chars_mask = tensor.matrix("features_mask")

cost  = poet.cost(chars, chars_mask)
cost.name = "sequence_log_likelihood"

model = Model(cost)
params = model.get_params()

for brick in model.get_top_bricks():
    brick.initialize()

cg = ComputationGraph(cost)
algorithm = GradientDescent(
    cost=cost, params=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)]))

monitor = TrainingDataMonitoring(
    variables=[cost],
    after_epoch=True,
    prefix="test")

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions=[
    monitor,
    Printing(),
    Write(poet = model.get_top_bricks()[0],after_epoch = True)
    ])

main_loop.run()