import ipdb

import numpy
from numpy import array

import theano
from theano import tensor

from blocks.algorithms import (GradientDescent, Scale,
                               RMSProp, Adam,
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
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                    DataStreamMonitoring)
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.filter import VariableFilter
from blocks.roles import INITIAL_STATE
from blocks.utils import shared_floatx_zeros

from fuel.transformers import Mapping
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream

from datasets.txt2hdf5 import Text

config.recursion_limit = 100000
floatX = theano.config.floatX

# Dictionaries
all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('A') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', '!', '?', '<UNK>', '\n'] +
             ['"', '"', ':', ';', '.', '-'] +
             ['(', ')', ' ', '<S>', '</S>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}

def _transpose(data):
    return tuple(array.T for array in data)

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

        # Skip connections to the end.
        sources = [name for name in transition.apply.states if 'states' in name]

        readout = Readout(
            readout_dim=alphabet_size,
            source_names=sources,
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
    def cost(self, chars, chars_mask = None, **kwargs):
        return aggregation.mean(self.generator.cost_matrix(
            chars, chars_mask, **kwargs).sum(), chars.shape[1])

class Write(SimpleExtension):
    """Make your poet write
    Parameters
    ----------
    chars : int Number of characters to generate
    """
    def __init__(self, poet, chars=100, **kwargs):
        super(Write, self).__init__(**kwargs)
        sample = ComputationGraph(poet.generator.generate(n_steps=chars, 
            batch_size=1, iterate=True)[-2])
        self.sample_fn = sample.get_theano_function()

    def do(self, callback_name, *args):
        outputs = self.sample_fn()[0]

        try:
            true_length = list(outputs).index(array(char2code['</S>'])) + 1
        except ValueError:
            true_length = len(outputs)
        outputs = outputs[:true_length]

        print "".join(code2char[code[0]] for code in outputs)

dimension = 800
poet = Poet(dimension = dimension,
            depth = 3,
            alphabet_size = len(char2code),
            name="poet")

poet.weights_init = IsotropicGaussian(0.05)
poet.biases_init = Constant(0.0)

dataset = Text('poet/datasets/input.txt', ('train',))

batch_size = 100

data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            dataset.num_examples, batch_size))
data_stream = Mapping(data_stream, _transpose)

test_dataset = Text('poet/datasets/input.txt', ('valid',))

test_stream = DataStream.default_stream(
            test_dataset, iteration_scheme=SequentialScheme(
            test_dataset.num_examples, batch_size))
test_stream = Mapping(test_stream, _transpose)

# Build the cost computation graph
chars = tensor.lmatrix("features")

states = poet.generator.transition.transition.apply.outputs
states = {name: shared_floatx_zeros((batch_size, dimesion))
          for name in states}

cost  = poet.cost(chars, **states)
cost.name = "sequence_log_likelihood"

model = Model(cost)
params = model.parameters

for brick in model.get_top_bricks():
    brick.initialize()

cg = ComputationGraph(cost)
parameters = cg.parameters

def make_regex(name_state = ""):
    return '[a-zA-Z_]*'+ name_state +'_final_value'

# Update the initial values with the last state
extra_updates = []
for name, var in states.items():
    update = VariableFilter(theano_name_regex=make_regex(name)
                    )(cg.auxiliary_variables)[0]
    extra_updates.append((var, update))

algorithm = GradientDescent(
    cost=cost, parameters=parameters,
    step_rule=CompositeRule([StepClipping(10.0), Adam(0.005)]))

algorithm.add_updates(extra_updates)

train_monitor = TrainingDataMonitoring(
    variables=[cost],
    every_n_batches = 100,
    prefix="train")

test_monitor = DataStreamMonitoring(
    variables = [cost],
    data_stream = test_stream,
    after_epoch = False,
    every_n_batches = 1000,
    prefix = "test")

extensions = extensions=[
    train_monitor,
    test_monitor,
    Printing(every_n_batches = 10),
    Write(poet = model.get_top_bricks()[0],
     every_n_batches = 10, chars = 2000),
    FinishAfter(after_n_batches = 2000)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=data_stream,
    algorithm=algorithm,
    extensions = extensions)

main_loop.run()