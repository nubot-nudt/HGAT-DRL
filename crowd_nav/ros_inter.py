#!/usr/bin/env python
import logging
import argparse
import importlib.util
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.policy.reward_estimate import Reward_Estimator
import rospy
from crowd_sim.envs.utils.state import ObservableState, FullState, JointState, ObstacleState, WallState
from sgdqn_common.msg import ObserveInfo, ActionCmd
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from dynamic_reconfigure.server import Server
from sgdqn_common.cfg import GoalConfig
class sgdqn_planner:
    def init(self):
        self.robot_policy = None
        self.peds_policy = None
        self.cur_state = None
        self.goal_x = 0.0
        self.goal_y = 4.0
        rospy.init_node('sgdqn_planner_node', anonymous=True)


    def start(self):
        srv = Server(GoalConfig, self.callback)
        rospy.Subscriber("observeInfo", ObserveInfo, self.state_callback)
        # self.human_vel_pub = rospy.Publisher('human_vel_cmd', VelInfo, queue_size=10)
        self.robot_action_pub = rospy.Publisher('robot_action_cmd', ActionCmd, queue_size=10)
        rospy.spin()

    def callback(self, ros_config, level):
        self.goal_x = ros_config.goal_x
        self.goal_y = ros_config.goal_y
        rospy.loginfo("""Reconfigure Request: {goal_x}, {goal_y}""".format(**ros_config))
        return ros_config

    def configure(self):
        self.robot_policy = policy_factory['tree_search_rl']()
        self.peds_policy = policy_factory['centralized_orca']()


    def load_policy_model(self, args):
        level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S")
        device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
        logging.info('Using device: %s', device)

        if args.model_dir is not None:
            if args.config is not None:
                config_file = args.config
            else:
                config_file = os.path.join(args.model_dir, 'config.py')
            if args.il:
                model_weights = os.path.join(args.model_dir, 'il_model.pth')
                logging.info('Loaded IL weights')
            elif args.rl:
                if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                    model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
                else:
                    print(os.listdir(args.model_dir))
                    model_weights = os.path.join(args.model_dir, sorted(os.listdir(args.model_dir))[-1])
                logging.info('Loaded RL weights')
            else:
                model_weights = os.path.join(args.model_dir, 'best_val.pth')
                logging.info('Loaded RL weights with best VAL')

        else:
            config_file = args.config

        spec = importlib.util.spec_from_file_location('config', config_file)
        if spec is None:
            parser.error('Config file not found.')
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

        # configure policy
        policy_config = config.PolicyConfig(args.debug)
        if policy_config.name == 'rgcn_rl':
            policy_config.gnn_model = args.gnn
        policy = policy_factory[policy_config.name]()
        reward_estimator = Reward_Estimator()
        env_config = config.EnvConfig(args.debug)
        reward_estimator.configure(env_config)
        policy.reward_estimator = reward_estimator
        if args.planning_depth is not None:
            policy_config.model_predictive_rl.do_action_clip = True
            policy_config.model_predictive_rl.planning_depth = args.planning_depth
        if args.planning_width is not None:
            policy_config.model_predictive_rl.do_action_clip = True
            policy_config.model_predictive_rl.planning_width = args.planning_width
        if args.sparse_search:
            policy_config.model_predictive_rl.sparse_search = True

        policy.configure(policy_config, device)
        if policy.trainable:
            if args.model_dir is None:
                parser.error('Trainable policy must be specified with a model weights directory')
            policy.load_model(model_weights)

        # configure environment
        env_config = config.EnvConfig(args.debug)

        if args.human_num is not None:
            env_config.sim.human_num = args.human_num
        env = gym.make('CrowdSim-v0')
        env.configure(env_config)

        if args.square:
            env.test_scenario = 'square_crossing'
        if args.circle:
            env.test_scenario = 'circle_crossing'
        if args.test_scenario is not None:
            env.test_scenario = args.test_scenario

        # for continous action
        robot = Robot(env_config, 'robot')
        env.set_robot(robot)
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high
        min_action = env.action_space.low
        if policy.name == 'TD3RL' or policy.name == 'RGCNRL':
            policy.set_action(action_dim, max_action, min_action)
        self.robot_policy = policy
        policy.set_v_pref(1.0)
        self.robot_policy.set_time_step(env.time_step)
        train_config = config.TrainConfig(args.debug)
        epsilon_end = train_config.train.epsilon_end
        if not isinstance(self.robot_policy, ORCA):
            self.robot_policy.set_epsilon(epsilon_end)

        policy.set_phase(args.phase)
        policy.set_device(device)

        # set safety space for ORCA in non-cooperative simulation
        if isinstance(self.robot_policy, ORCA):
            self.robot_policy.safety_space = args.safety_space
            logging.info('ORCA agent buffer: %f', self.robot_policy.safety_space)

    def state_callback(self, observe_info):
        robot_state = observe_info.robot_state
        robot_state.goal_x = self.goal_x
        robot_state.goal_y = self.goal_y
        robot_full_state = FullState(robot_state.pos_x, robot_state.pos_y, robot_state.vel_x, robot_state.vel_y,
                                     robot_state.radius, robot_state.goal_x, robot_state.goal_y, robot_state.vmax,
                                     robot_state.theta)
        peds_full_state = [ObservableState(ped_state.pos_x, ped_state.pos_y, ped_state.vel_x, ped_state.vel_y,
                                     ped_state.radius) for ped_state in observe_info.ped_states]
        obstacle_states = [ObstacleState(disc_obstacle.pos_x, disc_obstacle.pos_y, disc_obstacle.radius)
                          for disc_obstacle in observe_info.disc_states]
        disc_num = len(obstacle_states)
        wall_states = [WallState(wall_obstacle.start_x, wall_obstacle.start_y, wall_obstacle.end_x,
                                 wall_obstacle.end_y) for wall_obstacle in observe_info.line_states]
        wall_num = len(wall_states)
        if self.robot_policy.name == 'TD3RL':
            self.robot_policy.set_obstacle_num(disc_num, wall_num)
        observable_states = (peds_full_state, obstacle_states, wall_states)
        self.cur_state = JointState(robot_full_state, observable_states)

        action_cmd = ActionCmd()

        dis = np.sqrt((robot_full_state.px - robot_full_state.gx)**2 + (robot_full_state.py - robot_full_state.gy)**2)
        if dis < 0.5:
            action_cmd.stop = True
            action_cmd.vel_x = robot_state.vel_x
            action_cmd.vel_y = robot_state.vel_y
            action_cmd.acc_l = 0
            action_cmd.acc_r = 0
        else:
            print("state callback")
            action_cmd.stop = False
            robot_action, robot_action_index = self.robot_policy.predict(self.cur_state)
            print('robot_action', robot_action.al, robot_action.ar)
            action_cmd.acc_l = robot_action.al
            action_cmd.acc_r = robot_action.ar
            action_cmd.vel_x = robot_state.vel_x
            action_cmd.vel_y = robot_state.vel_y
            action_cmd.stamp = observe_info.stamp
        self.robot_action_pub.publish(action_cmd)
        # human_actions = self.peds_policy.predict(peds_full_state)
        #
        # test_action = ActionXY(0.0, 0.0)
        # robot_vel = AgentVel()
        # robot_vel.vel_x = test_action.vx
        # robot_vel.vel_y = test_action.vy
        # vel_infos = VelInfo()
        # vel_infos.vel_info.append(robot_vel)
        # # human policy
        # for human_action in human_actions:
        #     human_vel = AgentVel()
        #     human_vel.vel_x = human_action.vx
        #     human_vel.vel_y = human_action.vy
        #     vel_infos.vel_info.append(human_vel)
        # self.human_vel_pub.publish(vel_infos)

    def compute_observation(self, full_states):
        observation_states = [full_state.get_observable_state() for full_state in full_states]
        return observation_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='td3_rl')
    parser.add_argument('--gnn', type=str, default='transformer')
    parser.add_argument('-m', '--model_dir', type=str, default='data/final_data/transformer_test/transformer/5')#None
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=None)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    sys_args = parser.parse_args()
    try:
        planner = sgdqn_planner()
        planner.init()
        planner.configure()
        planner.load_policy_model(sys_args)
        planner.start()
    except rospy.ROSException:
        pass
