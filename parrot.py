import numpy
from numpy import array
import ipdb

import theano
from theano import tensor

from blocks.bricks import Tanh, Initializable
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, LSTM

from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.config import config
from blocks.graph import ComputationGraph
from fuel.transformers import Mapping, Batch, Padding, Filter
from fuel.datasets import OneBillionWord, TextFile
from fuel.schemes import ConstantScheme
from blocks.dump import load_parameter_values
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation

from blocks.extensions import FinishAfter, Timing
from blocks.extensions.saveload import Checkpoint

from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter
from blocks.utils import named_copy, dict_union

from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import SimpleExtension

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

class Parrot(Initializable):
    """Parrot is your favourite text-generation pet.
    Parrot should learn your favourite topics, and talk to you about them.
    """
    def __init__(self, dimension, alphabet_size, **kwargs):
        super(Parrot, self).__init__(**kwargs)

        lookup = LookupTable(alphabet_size, dimension)

        transition = LSTM(
            dim=dimension, name="transition")

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

class Sample(SimpleExtension):
    """Make your parrot talk
    Parameters
    ----------
    chars : int Number of characters to generate
    """
    def __init__(self, parrot, chars=100, **kwargs):
        super(Sample, self).__init__(**kwargs)
        steps = 300
        sample = ComputationGraph(parrot.generator.generate(n_steps=steps, 
            batch_size=1, iterate=True))
        self.sample_fn = sample.get_theano_function()

    def do(self, callback_name, *args):
        _1, _2, outputs, _3 = self.sample_fn()

        try:
            true_length = list(outputs).index(array(char2code['</S>'])) + 1
        except ValueError:
            true_length = len(outputs)
        outputs = outputs[:true_length]

        print "".join(code2char[code[0]] for code in outputs)


parrot = Parrot(dimension = 2000, alphabet_size = len(char2code), name="parrot")
parrot.weights_init = IsotropicGaussian(0.1)
parrot.biases_init = Constant(0.0)

dataset_options = dict(dictionary=char2code, level="character",
                               preprocess=_lower)

#dataset = TextFile(data_path, **dataset_options)
dataset = OneBillionWord("training", [99], **dataset_options)

data_stream = dataset.get_example_stream()
data_stream = Filter(data_stream, _filter_long)
data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10))
data_stream = Padding(data_stream)
data_stream = Mapping(data_stream, _transpose)

# Build the cost computation graph
chars = tensor.lmatrix("features")
chars_mask = tensor.matrix("features_mask")

cost  = parrot.cost(chars, chars_mask)
cost.name = "sequence_log_likelihood"

model = Model(cost)
params = model.get_params()

for brick in model.get_top_bricks():
    brick.initialize()

cg = ComputationGraph(cost)
algorithm = GradientDescent(
    cost=cost, params=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)]))

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions=[Sample(parrot = model.get_top_bricks()[0],every_n_batches = 100)])

main_loop.run()