from flymazerl.agents.qlearning import de_i
from flymazerl.agents.qlearning import de_lt
from flymazerl.agents.qlearning import df_i
from flymazerl.agents.qlearning import df_i_os
from flymazerl.agents.qlearning import df_lt
from flymazerl.agents.qlearning import df_lt_os
from flymazerl.agents.qlearning import esa
from flymazerl.agents.qlearning import f_i
from flymazerl.agents.qlearning import f_i_hv
from flymazerl.agents.qlearning import f_i_os
from flymazerl.agents.qlearning import f_lt
from flymazerl.agents.qlearning import f_lt_hv
from flymazerl.agents.qlearning import f_lt_os
from flymazerl.agents.qlearning import f_rf
from flymazerl.agents.qlearning import i
from flymazerl.agents.qlearning import i_hv
from flymazerl.agents.qlearning import i_os
from flymazerl.agents.qlearning import lt
from flymazerl.agents.qlearning import lt_hv
from flymazerl.agents.qlearning import lt_os
from flymazerl.agents.qlearning import rf
from flymazerl.agents.qlearning import sa

from flymazerl.agents.qlearning.de_i import (DEIQLearner, DEIQLearnerHM,
                                             DEIQLearnerHM_acceptreject,
                                             DEIQLearner_acceptreject,
                                             DEIQLearner_egreedy,
                                             DEIQLearner_esoftmax,
                                             DEIQLearner_softmax,)
from flymazerl.agents.qlearning.de_lt import (DELTQLearner, DELTQLearnerHM,
                                              DELTQLearnerHM_acceptreject,
                                              DELTQLearner_acceptreject,
                                              DELTQLearner_egreedy,
                                              DELTQLearner_esoftmax,
                                              DELTQLearner_softmax,)
from flymazerl.agents.qlearning.df_i import (DFIQLearner, DFIQLearnerHM,
                                             DFIQLearnerHM_acceptreject,
                                             DFIQLearner_acceptreject,
                                             DFIQLearner_egreedy,
                                             DFIQLearner_esoftmax,
                                             DFIQLearner_softmax,)
from flymazerl.agents.qlearning.df_i_os import (DFOSQLearner,
                                                DFOSQLearner_acceptreject,
                                                DFOSQLearner_egreedy,
                                                DFOSQLearner_esoftmax,
                                                DFOSQLearner_softmax,)
from flymazerl.agents.qlearning.df_lt import (DFLTQLearner, DFLTQLearnerHM,
                                              DFLTQLearnerHM_acceptreject,
                                              DFLTQLearner_acceptreject,
                                              DFLTQLearner_egreedy,
                                              DFLTQLearner_esoftmax,
                                              DFLTQLearner_softmax,)
from flymazerl.agents.qlearning.df_lt_os import (DFOSCQLearner,
                                                 DFOSCQLearner_acceptreject,
                                                 DFOSCQLearner_egreedy,
                                                 DFOSCQLearner_esoftmax,
                                                 DFOSCQLearner_softmax,)
from flymazerl.agents.qlearning.esa import (ESARSALearner, ESARSALearnerHM,
                                            ESARSALearnerHM_acceptreject,
                                            ESARSALearner_acceptreject,
                                            ESARSALearner_egreedy,
                                            ESARSALearner_esoftmax,
                                            ESARSALearner_softmax,)
from flymazerl.agents.qlearning.f_i import (FIQLearner, FIQLearnerHM,
                                            FIQLearnerHM_acceptreject,
                                            FIQLearner_acceptreject,
                                            FIQLearner_egreedy,
                                            FIQLearner_esoftmax,
                                            FIQLearner_softmax,)
from flymazerl.agents.qlearning.f_i_hv import (FHQLearner,
                                               FHQLearner_acceptreject,
                                               FHQLearner_egreedy,
                                               FHQLearner_esoftmax,
                                               FHQLearner_softmax,)
from flymazerl.agents.qlearning.f_i_os import (FOSQLearner,
                                               FOSQLearner_acceptreject,
                                               FOSQLearner_egreedy,
                                               FOSQLearner_esoftmax,
                                               FOSQLearner_softmax,)
from flymazerl.agents.qlearning.f_lt import (FLTQLearner, FLTQLearnerHM,
                                             FLTQLearnerHM_acceptreject,
                                             FLTQLearner_acceptreject,
                                             FLTQLearner_egreedy,
                                             FLTQLearner_esoftmax,
                                             FLTQLearner_softmax,)
from flymazerl.agents.qlearning.f_lt_hv import (FHCQLearner,
                                                FHCQLearner_acceptreject,
                                                FHCQLearner_egreedy,
                                                FHCQLearner_esoftmax,
                                                FHCQLearner_softmax,)
from flymazerl.agents.qlearning.f_lt_os import (FOSCQLearner,
                                                FOSCQLearner_acceptreject,
                                                FOSCQLearner_egreedy,
                                                FOSCQLearner_esoftmax,
                                                FOSCQLearner_softmax,)
from flymazerl.agents.qlearning.f_rf import (FRPEFreeLearner,
                                             FRPEFreeLearnerHM,
                                             FRPEFreeLearnerHM_acceptreject,
                                             FRPEFreeLearner_acceptreject,
                                             FRPEFreeLearner_egreedy,
                                             FRPEFreeLearner_esoftmax,
                                             FRPEFreeLearner_softmax,)
from flymazerl.agents.qlearning.i import (IQLearner, IQLearnerHM,
                                          IQLearnerHM_acceptreject,
                                          IQLearner_acceptreject,
                                          IQLearner_egreedy,
                                          IQLearner_esoftmax,
                                          IQLearner_softmax,)
from flymazerl.agents.qlearning.i_hv import (HQLearner, HQLearner_acceptreject,
                                             HQLearner_egreedy,
                                             HQLearner_esoftmax,
                                             HQLearner_softmax,)
from flymazerl.agents.qlearning.i_os import (OSQLearner,
                                             OSQLearner_acceptreject,
                                             OSQLearner_egreedy,
                                             OSQLearner_esoftmax,
                                             OSQLearner_softmax,)
from flymazerl.agents.qlearning.lt import (LTQLearner, LTQLearnerHM,
                                           LTQLearnerHM_acceptreject,
                                           LTQLearner_acceptreject,
                                           LTQLearner_egreedy,
                                           LTQLearner_esoftmax,
                                           LTQLearner_softmax,)
from flymazerl.agents.qlearning.lt_hv import (HCQLearner,
                                              HCQLearner_acceptreject,
                                              HCQLearner_egreedy,
                                              HCQLearner_esoftmax,
                                              HCQLearner_softmax,)
from flymazerl.agents.qlearning.lt_os import (OSCQLearner,
                                              OSCQLearner_acceptreject,
                                              OSCQLearner_egreedy,
                                              OSCQLearner_esoftmax,
                                              OSCQLearner_softmax,)
from flymazerl.agents.qlearning.rf import (RPEFreeLearner, RPEFreeLearnerHM,
                                           RPEFreeLearnerHM_acceptreject,
                                           RPEFreeLearner_acceptreject,
                                           RPEFreeLearner_egreedy,
                                           RPEFreeLearner_esoftmax,
                                           RPEFreeLearner_softmax,)
from flymazerl.agents.qlearning.sa import (SARSALearner, SARSALearnerHM,
                                           SARSALearnerHM_acceptreject,
                                           SARSALearner_acceptreject,
                                           SARSALearner_egreedy,
                                           SARSALearner_esoftmax,
                                           SARSALearner_softmax,)

__all__ = ['DEIQLearner', 'DEIQLearnerHM', 'DEIQLearnerHM_acceptreject',
           'DEIQLearner_acceptreject', 'DEIQLearner_egreedy',
           'DEIQLearner_esoftmax', 'DEIQLearner_softmax', 'DELTQLearner',
           'DELTQLearnerHM', 'DELTQLearnerHM_acceptreject',
           'DELTQLearner_acceptreject', 'DELTQLearner_egreedy',
           'DELTQLearner_esoftmax', 'DELTQLearner_softmax', 'DFIQLearner',
           'DFIQLearnerHM', 'DFIQLearnerHM_acceptreject',
           'DFIQLearner_acceptreject', 'DFIQLearner_egreedy',
           'DFIQLearner_esoftmax', 'DFIQLearner_softmax', 'DFLTQLearner',
           'DFLTQLearnerHM', 'DFLTQLearnerHM_acceptreject',
           'DFLTQLearner_acceptreject', 'DFLTQLearner_egreedy',
           'DFLTQLearner_esoftmax', 'DFLTQLearner_softmax', 'DFOSCQLearner',
           'DFOSCQLearner_acceptreject', 'DFOSCQLearner_egreedy',
           'DFOSCQLearner_esoftmax', 'DFOSCQLearner_softmax', 'DFOSQLearner',
           'DFOSQLearner_acceptreject', 'DFOSQLearner_egreedy',
           'DFOSQLearner_esoftmax', 'DFOSQLearner_softmax', 'ESARSALearner',
           'ESARSALearnerHM', 'ESARSALearnerHM_acceptreject',
           'ESARSALearner_acceptreject', 'ESARSALearner_egreedy',
           'ESARSALearner_esoftmax', 'ESARSALearner_softmax', 'FHCQLearner',
           'FHCQLearner_acceptreject', 'FHCQLearner_egreedy',
           'FHCQLearner_esoftmax', 'FHCQLearner_softmax', 'FHQLearner',
           'FHQLearner_acceptreject', 'FHQLearner_egreedy',
           'FHQLearner_esoftmax', 'FHQLearner_softmax', 'FIQLearner',
           'FIQLearnerHM', 'FIQLearnerHM_acceptreject',
           'FIQLearner_acceptreject', 'FIQLearner_egreedy',
           'FIQLearner_esoftmax', 'FIQLearner_softmax', 'FLTQLearner',
           'FLTQLearnerHM', 'FLTQLearnerHM_acceptreject',
           'FLTQLearner_acceptreject', 'FLTQLearner_egreedy',
           'FLTQLearner_esoftmax', 'FLTQLearner_softmax', 'FOSCQLearner',
           'FOSCQLearner_acceptreject', 'FOSCQLearner_egreedy',
           'FOSCQLearner_esoftmax', 'FOSCQLearner_softmax', 'FOSQLearner',
           'FOSQLearner_acceptreject', 'FOSQLearner_egreedy',
           'FOSQLearner_esoftmax', 'FOSQLearner_softmax', 'FRPEFreeLearner',
           'FRPEFreeLearnerHM', 'FRPEFreeLearnerHM_acceptreject',
           'FRPEFreeLearner_acceptreject', 'FRPEFreeLearner_egreedy',
           'FRPEFreeLearner_esoftmax', 'FRPEFreeLearner_softmax', 'HCQLearner',
           'HCQLearner_acceptreject', 'HCQLearner_egreedy',
           'HCQLearner_esoftmax', 'HCQLearner_softmax', 'HQLearner',
           'HQLearner_acceptreject', 'HQLearner_egreedy', 'HQLearner_esoftmax',
           'HQLearner_softmax', 'IQLearner', 'IQLearnerHM',
           'IQLearnerHM_acceptreject', 'IQLearner_acceptreject',
           'IQLearner_egreedy', 'IQLearner_esoftmax', 'IQLearner_softmax',
           'LTQLearner', 'LTQLearnerHM', 'LTQLearnerHM_acceptreject',
           'LTQLearner_acceptreject', 'LTQLearner_egreedy',
           'LTQLearner_esoftmax', 'LTQLearner_softmax', 'OSCQLearner',
           'OSCQLearner_acceptreject', 'OSCQLearner_egreedy',
           'OSCQLearner_esoftmax', 'OSCQLearner_softmax', 'OSQLearner',
           'OSQLearner_acceptreject', 'OSQLearner_egreedy',
           'OSQLearner_esoftmax', 'OSQLearner_softmax', 'RPEFreeLearner',
           'RPEFreeLearnerHM', 'RPEFreeLearnerHM_acceptreject',
           'RPEFreeLearner_acceptreject', 'RPEFreeLearner_egreedy',
           'RPEFreeLearner_esoftmax', 'RPEFreeLearner_softmax', 'SARSALearner',
           'SARSALearnerHM', 'SARSALearnerHM_acceptreject',
           'SARSALearner_acceptreject', 'SARSALearner_egreedy',
           'SARSALearner_esoftmax', 'SARSALearner_softmax', 'de_i', 'de_lt',
           'df_i', 'df_i_os', 'df_lt', 'df_lt_os', 'esa', 'f_i', 'f_i_hv',
           'f_i_os', 'f_lt', 'f_lt_hv', 'f_lt_os', 'f_rf', 'i', 'i_hv', 'i_os',
           'lt', 'lt_hv', 'lt_os', 'rf', 'sa']
