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
from crowd_nav.policy.actor_critic_guard import Safe_Explorer

def main(args):
    # configure logging and device
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
    if policy_config.name == 'rgcn_rl' or 'rgcn_acg_rl':
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

    if args.human_num is not None:
        env_config.sim.human_num = args.human_num
        policy_config.gat.human_num = args.human_num

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
    if policy.name == 'TD3RL' or 'RGCNRL' or 'RGCN_ACG_RL' :
        policy.set_action(action_dim, max_action, min_action)
    robot.time_step = env.time_step
    robot.set_policy(policy)
    if policy.name == 'RGCN_ACG_RL':
        explorer = Safe_Explorer(env, robot, device, None, gamma=0.95)
    else:
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
    env.set_phase(3)
    policy.set_env(env)
    robot.print_info()
    vel_rec = []
    if args.visualize:
        if robot.policy.name in ['tree_search_rl']:
            policy.model[2].eval()
        rewards = []
        actions = []
        constraints = []
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())

        while not done:
            action, action_index = robot.act(ob)
            if policy.name =='RGCN_ACG_RL':
                ob, _, constraint, done, info = env.step(action)
            else:
                ob, _, done, info = env.step(action)
            if isinstance(info, Timeout):
                _ = _ - 0.25
            rewards.append(_)
            constraints.append(constraint)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
            actions.append(action)
            last_velocity = np.array(robot.get_velocity())
            vel_rec.append(last_velocity)
        last_velocity = np.array(robot.get_velocity())
        vel_rec.append(last_velocity)
        gamma = 0.95
        cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
             * reward for t, reward in enumerate(rewards)])
        cumulative_constraint = sum([pow(gamma, t * robot.time_step * robot.v_pref)
             * constraint for t, constriant in enumerate(constraints)])
        positions = []
        velocity_left_rec = []
        velocity_right_rec = []
        velocity_rec = []
        rotation_rec = []
        action_left_rec = []
        action_right_rec = []
        for i in range(len(vel_rec) - 1):
            if i % 1 == 0:
                positions.append(i * robot.time_step)
                vel = vel_rec[i]
                action = actions[i]
                if robot.kinematics is 'unicycle':
                    velocity_left_rec.append(vel[0])
                    velocity_right_rec.append((vel[1]))
                    velocity_rec.append((vel[0] + vel[1]) * 0.5)
                    rotation_rec.append((vel[1] - vel[0]) / (2 * robot.radius))
                elif robot.kinematics is 'holonomic':
                    velocity_rec.append(vel[0])
                    rotation_rec.append(vel[1])
                elif robot.kinematics is 'differential':
                    velocity_left_rec.append(vel[0])
                    velocity_right_rec.append((vel[1]))
                    velocity_rec.append((vel[0] + vel[1]) * 0.5)
                    rotation_rec.append((vel[1] - vel[0]) / (2 * robot.radius))
                    action_left_rec.append(action.al)
                    action_right_rec.append(action.ar)
        if args.traj:
            env.render('traj', args.video_file)
        else:
            if args.video_dir is not None:
                if policy_config.name == 'gcn':
                    args.video_file = os.path.join(args.video_dir, policy_config.name + '_' + policy_config.gcn.similarity_function)
                else:
                    args.video_file = os.path.join(args.video_dir, policy_config.name)
                args.video_file = args.video_file + '_' + args.phase + '_' + str(args.test_case) + '.mp4'
            env.render('video', args.video_file)
        logging.info('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f cumulative '
                     'constraint is %f ', env.global_time, info, cumulative_reward, cumulative_constraint)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))

    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)
        if args.plot_test_scenarios_hist:
            test_angle_seeds = np.array(env.test_scene_seeds)
            b = [i * 0.01 for i in range(101)]
            n, bins, patches = plt.hist(test_angle_seeds, b, facecolor='g')
            plt.savefig(os.path.join(args.model_dir, 'test_scene_hist.png'))
            plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='rgcn_acg_rl')
    parser.add_argument('--gnn', type=str, default='rgcn')
    parser.add_argument('-m', '--model_dir', type=str, default='data/output')#None
    # parser.add_argument('--policy', type=str, default='td3_rl')
    # parser.add_argument('--gnn', type=str, default='transformer')
    # parser.add_argument('-m', '--model_dir', type=str, default='data/final_data/transformer_test/transformer/4')#None
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=27)
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
