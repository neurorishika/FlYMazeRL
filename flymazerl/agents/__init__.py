# __init__.py

from flymazerl.agents import base
from flymazerl.agents import classical
from flymazerl.agents import neuralnetworks
from flymazerl.agents import phenomenological

from flymazerl.agents.base import (FlYMazeAgent, RandomAgent,)
from flymazerl.agents.classical import (CQLearner, CQLearner_acceptreject,
                                        CQLearner_egreedy, CQLearner_esoftmax,
                                        CQLearner_softmax, DECQLearner,
                                        DECQLearner_acceptreject,
                                        DECQLearner_egreedy,
                                        DECQLearner_esoftmax,
                                        DECQLearner_softmax, DEQLearner,
                                        DEQLearner_acceptreject,
                                        DEQLearner_egreedy,
                                        DEQLearner_esoftmax,
                                        DEQLearner_softmax, DFCQLearner,
                                        DFCQLearner_acceptreject,
                                        DFCQLearner_egreedy,
                                        DFCQLearner_esoftmax,
                                        DFCQLearner_softmax, DFOSCQLearner,
                                        DFOSCQLearner_acceptreject,
                                        DFOSCQLearner_egreedy,
                                        DFOSCQLearner_esoftmax,
                                        DFOSCQLearner_softmax, DFOSQLearner,
                                        DFOSQLearner_acceptreject,
                                        DFOSQLearner_egreedy,
                                        DFOSQLearner_esoftmax,
                                        DFOSQLearner_softmax, DFQLearner,
                                        DFQLearner_acceptreject,
                                        DFQLearner_egreedy,
                                        DFQLearner_esoftmax,
                                        DFQLearner_softmax, DQLearner,
                                        DQLearner_egreedy, DQLearner_esoftmax,
                                        DQLearner_softmax, ESARSALearner,
                                        ESARSALearner_acceptreject,
                                        ESARSALearner_egreedy,
                                        ESARSALearner_esoftmax,
                                        ESARSALearner_softmax, FCQLearner,
                                        FCQLearner_acceptreject,
                                        FCQLearner_egreedy,
                                        FCQLearner_esoftmax,
                                        FCQLearner_softmax, FHCQLearner,
                                        FHCQLearner_acceptreject,
                                        FHCQLearner_egreedy,
                                        FHCQLearner_esoftmax,
                                        FHCQLearner_softmax, FHQLearner,
                                        FHQLearner_acceptreject,
                                        FHQLearner_egreedy,
                                        FHQLearner_esoftmax,
                                        FHQLearner_softmax, FOSCQLearner,
                                        FOSCQLearner_acceptreject,
                                        FOSCQLearner_egreedy,
                                        FOSCQLearner_esoftmax,
                                        FOSCQLearner_softmax, FOSQLearner,
                                        FOSQLearner_acceptreject,
                                        FOSQLearner_egreedy,
                                        FOSQLearner_esoftmax,
                                        FOSQLearner_softmax, FQLearner,
                                        FQLearner_acceptreject,
                                        FQLearner_egreedy, FQLearner_esoftmax,
                                        FQLearner_softmax,
                                        ForgettingRewardLearner,
                                        ForgettingRewardLearner_acceptreject,
                                        ForgettingRewardLearner_egreedy,
                                        ForgettingRewardLearner_esoftmax,
                                        ForgettingRewardLearner_softmax,
                                        HCQLearner, HCQLearner_acceptreject,
                                        HCQLearner_egreedy,
                                        HCQLearner_esoftmax,
                                        HCQLearner_softmax, HQLearner,
                                        HQLearner_acceptreject,
                                        HQLearner_egreedy, HQLearner_esoftmax,
                                        HQLearner_softmax, IQLearner,
                                        IQLearner_acceptreject,
                                        IQLearner_egreedy, IQLearner_esoftmax,
                                        IQLearner_softmax, OSCQLearner,
                                        OSCQLearner_acceptreject,
                                        OSCQLearner_egreedy,
                                        OSCQLearner_esoftmax,
                                        OSCQLearner_softmax, OSQLearner,
                                        OSQLearner_acceptreject,
                                        OSQLearner_egreedy,
                                        OSQLearner_esoftmax,
                                        OSQLearner_softmax, RewardLearner,
                                        RewardLearner_acceptreject,
                                        RewardLearner_egreedy,
                                        RewardLearner_esoftmax,
                                        RewardLearner_softmax, SARSALearner,
                                        SARSALearner_acceptreject,
                                        SARSALearner_egreedy,
                                        SARSALearner_esoftmax,
                                        SARSALearner_softmax,)
from flymazerl.agents.neuralnetworks import (GFFNN, GQLearner, GRNNLearner,
                                             GRUNN, LSTMNN, VanillaRNN,)
from flymazerl.agents.phenomenological import (BayesianIdealObserver,
                                               CATIELearner,)

__all__ = ['BayesianIdealObserver', 'CATIELearner', 'CQLearner',
           'CQLearner_acceptreject', 'CQLearner_egreedy', 'CQLearner_esoftmax',
           'CQLearner_softmax', 'DECQLearner', 'DECQLearner_acceptreject',
           'DECQLearner_egreedy', 'DECQLearner_esoftmax',
           'DECQLearner_softmax', 'DEQLearner', 'DEQLearner_acceptreject',
           'DEQLearner_egreedy', 'DEQLearner_esoftmax', 'DEQLearner_softmax',
           'DFCQLearner', 'DFCQLearner_acceptreject', 'DFCQLearner_egreedy',
           'DFCQLearner_esoftmax', 'DFCQLearner_softmax', 'DFOSCQLearner',
           'DFOSCQLearner_acceptreject', 'DFOSCQLearner_egreedy',
           'DFOSCQLearner_esoftmax', 'DFOSCQLearner_softmax', 'DFOSQLearner',
           'DFOSQLearner_acceptreject', 'DFOSQLearner_egreedy',
           'DFOSQLearner_esoftmax', 'DFOSQLearner_softmax', 'DFQLearner',
           'DFQLearner_acceptreject', 'DFQLearner_egreedy',
           'DFQLearner_esoftmax', 'DFQLearner_softmax', 'DQLearner',
           'DQLearner_egreedy', 'DQLearner_esoftmax', 'DQLearner_softmax',
           'ESARSALearner', 'ESARSALearner_acceptreject',
           'ESARSALearner_egreedy', 'ESARSALearner_esoftmax',
           'ESARSALearner_softmax', 'FCQLearner', 'FCQLearner_acceptreject',
           'FCQLearner_egreedy', 'FCQLearner_esoftmax', 'FCQLearner_softmax',
           'FHCQLearner', 'FHCQLearner_acceptreject', 'FHCQLearner_egreedy',
           'FHCQLearner_esoftmax', 'FHCQLearner_softmax', 'FHQLearner',
           'FHQLearner_acceptreject', 'FHQLearner_egreedy',
           'FHQLearner_esoftmax', 'FHQLearner_softmax', 'FOSCQLearner',
           'FOSCQLearner_acceptreject', 'FOSCQLearner_egreedy',
           'FOSCQLearner_esoftmax', 'FOSCQLearner_softmax', 'FOSQLearner',
           'FOSQLearner_acceptreject', 'FOSQLearner_egreedy',
           'FOSQLearner_esoftmax', 'FOSQLearner_softmax', 'FQLearner',
           'FQLearner_acceptreject', 'FQLearner_egreedy', 'FQLearner_esoftmax',
           'FQLearner_softmax', 'FlYMazeAgent', 'ForgettingRewardLearner',
           'ForgettingRewardLearner_acceptreject',
           'ForgettingRewardLearner_egreedy',
           'ForgettingRewardLearner_esoftmax',
           'ForgettingRewardLearner_softmax', 'GFFNN', 'GQLearner',
           'GRNNLearner', 'GRUNN', 'HCQLearner', 'HCQLearner_acceptreject',
           'HCQLearner_egreedy', 'HCQLearner_esoftmax', 'HCQLearner_softmax',
           'HQLearner', 'HQLearner_acceptreject', 'HQLearner_egreedy',
           'HQLearner_esoftmax', 'HQLearner_softmax', 'IQLearner',
           'IQLearner_acceptreject', 'IQLearner_egreedy', 'IQLearner_esoftmax',
           'IQLearner_softmax', 'LSTMNN', 'OSCQLearner',
           'OSCQLearner_acceptreject', 'OSCQLearner_egreedy',
           'OSCQLearner_esoftmax', 'OSCQLearner_softmax', 'OSQLearner',
           'OSQLearner_acceptreject', 'OSQLearner_egreedy',
           'OSQLearner_esoftmax', 'OSQLearner_softmax', 'RandomAgent',
           'RewardLearner', 'RewardLearner_acceptreject',
           'RewardLearner_egreedy', 'RewardLearner_esoftmax',
           'RewardLearner_softmax', 'SARSALearner',
           'SARSALearner_acceptreject', 'SARSALearner_egreedy',
           'SARSALearner_esoftmax', 'SARSALearner_softmax', 'VanillaRNN',
           'base', 'classical', 'neuralnetworks', 'phenomenological']
