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
from crowd_sim.envs.utils.action import ActionRot, ActionXY, ActionDiff
from crowd_nav.utils.a_star import Astar

import rospy
import numpy as np
#from test_py_ros.msg import test
from std_msgs.msg import String
from sgdqn_common.msg import ObserveInfo, RobotState, PedState
from sgdqn_common.srv import TebCrowdSim, TebCrowdSimRequest, TebCrowdSimResponse
import math
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import PolygonStamped, Point32 , Twist,PoseArray
from geometry_msgs.msg import Pose2D, Pose, Twist
def ob2req(ob, robot_state):
    tebCrowdSimInfo = TebCrowdSim()
    human_state, obstacle_state, line_obstacle = ob
    robot_pose, robot_velocity, robot_goal = generateRobotPose(robot_state)
    obstaclearray = generateObstacles(obstacle_state, human_state, line_obstacle)
    return robot_pose, robot_velocity, robot_goal, obstaclearray

def generateRobotPose(robot_state):
    robot_pose2d = Pose2D()
    robot_pose2d.x = robot_state.px
    robot_pose2d.y = robot_state.py
    robot_pose2d.theta = robot_state.theta
    robot_pose = robot_pose2d
    robot_velocity = Twist()
    robot_velocity.linear.x = (robot_state.vx + robot_state.vy) / 2.0
    robot_velocity.linear.y = 0
    robot_velocity.linear.z = 0
    robot_velocity.angular.z = (robot_state.vx - robot_state.vy) / (2.0 * robot_state.radius)
    robot_velocity.angular.y = 0
    robot_velocity.angular.x = 0
    robot_goal = Pose2D()
    robot_goal.x = robot_state.gx
    robot_goal.y = robot_state.gy
    robot_goal.theta = np.pi * 0.5
    return robot_pose, robot_velocity,robot_goal

def generateObstacles(obstacle_state, human_state, line_obstacle):
    obstacle_msg = ObstacleArrayMsg()
    obstacle_msg.header.stamp = rospy.Time.now()
    obstacle_msg.header.frame_id = "odom"  # CHANGE HERE: odom/map
    id = 0
    # Add point obstacle
    for obstacle in obstacle_state:
        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[-1].id = id
        obstacle_msg.obstacles[-1].polygon.points = [Point32()]
        obstacle_msg.obstacles[-1].polygon.points[0].x = obstacle.px
        obstacle_msg.obstacles[-1].polygon.points[0].y = obstacle.py
        obstacle_msg.obstacles[-1].polygon.points[0].z = 0
        obstacle_msg.obstacles[-1].radius = obstacle.radius
        id = id + 1

    for human in human_state:
        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[-1].id = id
        obstacle_msg.obstacles[-1].polygon.points = [Point32()]
        obstacle_msg.obstacles[-1].polygon.points[0].x = human.px
        obstacle_msg.obstacles[-1].polygon.points[0].y = human.py
        obstacle_msg.obstacles[-1].polygon.points[0].z = 0
        obstacle_msg.obstacles[-1].radius = human.radius
        obstacle_msg.obstacles[-1].velocities.twist.linear.x = human.vx
        obstacle_msg.obstacles[-1].velocities.twist.linear.y = human.vy
        obstacle_msg.obstacles[-1].velocities.twist.linear.z = 0
        obstacle_msg.obstacles[-1].velocities.twist.angular.x = 0
        obstacle_msg.obstacles[-1].velocities.twist.angular.y = 0
        obstacle_msg.obstacles[-1].velocities.twist.angular.z = 0
        id = id + 1

    for line in line_obstacle:
        obstacle_msg.obstacles.append(ObstacleMsg())
        obstacle_msg.obstacles[-1].id = id
        line_start = Point32()
        line_start.x = line.sx
        line_start.y = line.sy
        line_end = Point32()
        line_end.x = line.ex
        line_end.y = line.ey
        obstacle_msg.obstacles[-1].polygon.points = [line_start, line_end]
        id = id + 1
    return obstacle_msg

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
    env_config.env.time_step = 0.25
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
    astarplanner = Astar()
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    # for continous action
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low
    if policy.name == 'TD3RL' or policy.name == 'RGCNRL':
        policy.set_action(action_dim, max_action, min_action)
    robot.time_step = env.time_step
    print(robot.time_step)
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
    env.set_phase(env_config.sim.human_num)
    print(env_config.sim.human_num)
    print(sys_args.randomseed)
    policy.set_env(env)
    robot.print_info()
    vel_rec = []

    r = rospy.Rate(5)  # 10hz
    t = 0.0

    if True:
        rewards = []
        actions = []
        done = False
        ob = env.reset(args.phase, args.test_case)
        last_pos = np.array(robot.get_position())
        while not done:
            target_pos = astarplanner.set_state2((env.robot.get_full_state(), ob[0], ob[1], ob[2]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--gnn', type=str, default='rgcn')
    parser.add_argument('-m', '--model_dir', type=str, default='data/from_zirui/0605/gat/10/')#None
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--rl', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('-v', '--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('-c', '--test_case', type=int, default=26)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--human_num', type=int, default=10)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--randomseed', type=int, default=7)
    parser.add_argument('--sparse_search', default=False, action='store_true')
    sys_args = parser.parse_args()
    main(sys_args)