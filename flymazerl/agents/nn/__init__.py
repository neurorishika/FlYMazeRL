from flymazerl.agents.nn import ffnn
from flymazerl.agents.nn import rnn

from flymazerl.agents.nn.ffnn import (GFFNN, GQLearner,)
from flymazerl.agents.nn.rnn import (GRNNLearner, GRUNN, LSTMNN, VanillaRNN,)

__all__ = ['GFFNN', 'GQLearner', 'GRNNLearner', 'GRUNN', 'LSTMNN',
           'VanillaRNN', 'ffnn', 'rnn']
