from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.gcn import GCN
from crowd_nav.policy.model_predictive_rl import ModelPredictiveRL
from crowd_nav.policy.gat_predictive_rl import GatPredictiveRL
from crowd_nav.policy.tree_searchrl import TreeSearchRL
from crowd_nav.policy.TD3_rl import TD3RL
from crowd_nav.policy.rgcnrl import RGCNRL
from crowd_nav.policy.actor_critic_guard import RGCN_ACG_RL

policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['gcn'] = GCN
policy_factory['model_predictive_rl'] = ModelPredictiveRL
policy_factory['tree_search_rl'] = TreeSearchRL
policy_factory['td3_rl'] = TD3RL
policy_factory['rgcn_rl'] = RGCNRL
policy_factory['rgcn_acg_rl'] = RGCN_ACG_RL