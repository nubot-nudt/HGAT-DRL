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
from crowd_sim.envs.policy.socialforce import SocialForce
from crowd_nav.policy.reward_estimate import Reward_Estimator
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionRot
import pandas as pd
import xlwt

def test(human_num, obstacle_num, model_dir, args):
    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    config_file = os.path.join(model_dir, 'config.py')
    model_weights = os.path.join(model_dir, 'best_val.pth')
    logging.info('Loaded RL weights with best VAL')


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


    env_config.sim.human_num = human_num
    env_config.sim.obstacle_num = obstacle_num
    policy_config.gat.human_num = human_num

    # configure environment
    env_config = config.EnvConfig(args.debug)

    policy.configure(policy_config, device)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.load_model(model_weights)

    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    if args.square:
        env.test_scenario = 'square_crossing'
    if args.circle:
        env.test_scenario = 'circle_crossing'
    if args.test_scenario is not None:
        env.test_scenario = args.test_scenario

    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    # for continous action
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low
    if policy.name == 'TD3RL' or policy.name == 'RGCNRL':
        policy.set_action(action_dim, max_action, min_action)
    robot.time_step = env.time_step
    robot.set_policy(policy)
    explorer = Explorer(env, robot, device, None, gamma=0.95)

    train_config = config.TrainConfig(args.debug)
    epsilon_end = train_config.train.epsilon_end
    if not (isinstance(robot.policy, ORCA) or isinstance(robot.policy, SocialForce)):
        robot.policy.set_epsilon(epsilon_end)

    policy.set_phase(args.phase)
    policy.set_device(device)

    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = args.safety_space
        else:
            robot.policy.safety_space = args.safety_space
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)
    env.set_phase(10)
    policy.set_env(env)
    robot.print_info()
    vel_rec = []
    return explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)

def main(args):
    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    sheet_head_info = ['seed', 'human_num', 'obs_num','suc_rate', 'col_rate', 'nav_time', 'col_time', 'cum_reward', 'ave_return', 'discomfort', 'total_time']
    for i in range(len(sheet_head_info)):
        booksheet.write(0, i, sheet_head_info[i])

    count = 1
    if args.model_dir is not None:
        dirs = os.listdir(args.model_dir)
        for gnn_dir in dirs:
            if os.path.isdir(args.model_dir + '/' + gnn_dir):
                dirs_models = os.listdir(args.model_dir + '/' + gnn_dir)
                print(args.model_dir + '/' + gnn_dir)
                args.gnn = gnn_dir
                for dir_file in dirs_models:
                    model_dir = args.model_dir + '/' + gnn_dir + '/' + dir_file
                    for human_num in range(10, 11):
                        obstacle_num = 3
                        statistical = test(human_num, obstacle_num, model_dir, args)
                        booksheet.write(count, 0, dir_file)
                        booksheet.write(count, 1, human_num)
                        booksheet.write(count, 2, obstacle_num)
                        for i in range(len(statistical)):
                            booksheet.write(count, 3 + i, statistical[i])
                        count = count + 1
                        save_path = args.model_dir + '/result1.xls'
                        workbook.save(save_path)

                    for obstacle_num in range(10, 11):
                        human_num = 5
                        statistical = test(human_num, obstacle_num, model_dir, args)
                        booksheet.write(count, 0, dir_file)
                        booksheet.write(count, 1, human_num)
                        booksheet.write(count, 2, obstacle_num)
                        for i in range(len(statistical)):
                            booksheet.write(count, 3 + i, statistical[i])
                        count = count + 1
                        save_path = args.model_dir + '/result1.xls'
                        workbook.save(save_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--gnn', type=str, default='rgcn')
    parser.add_argument('-m', '--model_dir', type=str, default='/home/nubot1/workspace/2021TITS/crowd_nav/data/final_data')#None
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=10)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=5)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    sys_args = parser.parse_args()
    main(sys_args)
