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
from crowd_sim.envs.policy.socialforce import SocialForce

from crowd_sim.envs.policy.orca import ORCA
from crowd_nav.policy.reward_estimate import Reward_Estimator
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionRot, ActionXY

import rospy
import numpy as np
#from test_py_ros.msg import test
from std_msgs.msg import String
from sgdqn_common.msg import ObserveInfo, RobotState, PedState
import math
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import PolygonStamped, Point32 , Twist,PoseArray

def main(args):
    global vx,vy,w,v
    # configure logging and device
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)
    robot_pub = rospy.Publisher('/robot_state',RobotState,queue_size = 1)
    human_pub = rospy.Publisher('/test_optim_node/obstacles', ObstacleArrayMsg, queue_size=1)

    rospy.init_node("test_obstacle_msg")
    obstacle_msg = ObstacleArrayMsg()
    obstacle_msg.header.stamp = rospy.Time.now()
    obstacle_msg.header.frame_id = "odom"  # CHANGE HERE: odom/map
    robot_state = RobotState()

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

    r = rospy.Rate(5)  # 10hz
    t = 0.0

    while not rospy.is_shutdown():
        for i in range(1000):
            rewards = []
            actions = []
            done = False
            ob = env.reset(args.phase, args.test_case)
            while not done:

                velocity = rospy.wait_for_message('/velocity', Twist, timeout=None)
                # velocity = rospy.wait_for_message('/test_optim_node/teb_velocity',Twist,timeout = None)
                vx = velocity.linear.x
                vy = velocity.linear.y

                w = velocity.angular.z
                v = velocity.linear.z
                print(vx)
                print(vy)

                action = ActionXY(vx, vy) \
                    if robot.kinematics == 'holonomic' else ActionRot(v, w)
                # action, action_index = robot.act(ob)
                ob, _, done, info = env.step(action)
                # for i in range(len(ob)):
                #     obstacle_msg.obstacles[i].polygon.points[0].x = ob[i].px
                #     obstacle_msg.obstacles[i].polygon.points[0].y = ob[i].py

                robot_state.pos_x = robot.px
                robot_state.pos_y = robot.py
                robot_state.vel_x = robot.vx
                robot_state.vel_y = robot.vy
                robot_state.radius = robot.radius
                robot_state.vmax = 5.0
                robot_state.theta = robot.theta
                robot_state.goal_x = robot.gx
                robot_state.goal_y = robot.gy

                human_pub.publish(obstacle_msg)
                robot_pub.publish(robot_state)
                ob, _, done, info = env.step(action)
                if isinstance(info, Timeout):
                    _ = _ - 0.25
                rewards.append(_)
                current_pos = np.array(robot.get_position())
                logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
                last_pos = current_pos
                actions.append(action)
                last_velocity = np.array(robot.get_velocity())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='rgcnrl')
    parser.add_argument('--gnn', type=str, default='transformer')
    parser.add_argument('-m', '--model_dir', type=str, default='data/0604/transformer/7')#None
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


# import rospy
# from roscpp_tutorials.srv import TwoInts, TwoIntsResponse
# def callback(request):
#     response = TwoIntsResponse()
#     response.sum = request.a + request.b;
#     return response
#
# if __name__ == '__main__':
#     rospy.init_node("py_service_node")
#     rospy.Service("py_service", TwoInts, callback)
#     rospy.spin()