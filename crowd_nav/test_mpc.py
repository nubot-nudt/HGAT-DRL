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
    robot_goal.theta = np.pi
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

    obstacle_msg.obstacles.append(ObstacleMsg())
    obstacle_msg.obstacles[-1].id = id
    obstacle_msg.obstacles[-1].polygon.points = [Point32()]
    obstacle_msg.obstacles[-1].polygon.points[0].x = 0.0
    obstacle_msg.obstacles[-1].polygon.points[0].y = 0.5
    obstacle_msg.obstacles[-1].polygon.points[0].z = 0
    obstacle_msg.obstacles[-1].radius = 1.5
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
    rospy.init_node("crowd_sim_node")

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

    if args.visualize:
        rewards = []
        actions = []
        done = False
        ob = env.reset(args.phase, args.test_case)
        last_pos = np.array(robot.get_position())
        while not done:
            robot_pose, robot_velocity, robot_goal, obstacles = ob2req(ob, env.robot.get_full_state())
            rospy.wait_for_service('/test_mpc_crowd_sim_node/crowd_sim_info')
            try:
                # 创建服务的处理句柄,可以像调用函数一样，调用句柄
                teb_server = rospy.ServiceProxy('/test_mpc_crowd_sim_node/crowd_sim_info', TebCrowdSim)
                resp1 = teb_server(robot_pose, robot_velocity, robot_goal, obstacles)
                vx = resp1.velocity.linear.x
                dt = resp1.velocity.linear.y
                w = resp1.velocity.angular.z
                if robot.kinematics is 'differential':
                    vel_left = (vx - w * env.robot.radius)
                    vel_right = (vx + w * env.robot.radius)
                    if dt == 0.0:
                        print("dt is 0")
                        dt = 0.5
                    al = (vel_left - env.robot.v_left) / (0.5 * dt)
                    ar = (vel_right - env.robot.v_right) / (0.5 * dt)
                    action = ActionDiff(al, ar)
                elif robot.kinematics is 'unicycle':
                    theta = w * robot.time_step
                    action = ActionRot(vx, theta)
                else:
                    action = ActionXY(vx * np.cos(robot.theta), vy * np.sin(robot.theta))
                ob, _, done, info = env.step(action)
                rewards.append(_)
                current_pos = np.array(robot.get_position())
                logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
                last_pos = current_pos
                actions.append(action)
                last_velocity = np.array(robot.get_velocity())
                vel_rec.append(last_velocity)
                # 如果调用失败，可能会抛出rospy.ServiceException
            except rospy.ServiceException:
                print("Service call failed:")
        last_velocity = np.array(robot.get_velocity())
        vel_rec.append(last_velocity)
        gamma = 0.95
        cumulative_reward = sum([pow(gamma, t * robot.time_step * robot.v_pref)
                                 * reward for t, reward in enumerate(rewards)])
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
        #
        # plt.plot(positions, velocity_left_rec, color='green', marker='*', linestyle='solid')
        # plt.plot(positions, velocity_right_rec, color='magenta', marker='^', linestyle='solid')
        #
        # plt.xlabel("t(s)")
        # # plt.ylim((-250, 250))
        # plt.grid = True
        # plt.plot(positions, velocity_left_rec, color='green', marker='d', markersize=2, linestyle='solid',
        #          label="vel_l(m/s)")
        # plt.plot(positions, velocity_right_rec, color='magenta', marker='^', markersize=2, linestyle='solid',
        #          label="vel_r(m/s)")
        # plt.plot(positions, velocity_rec, color='blue', marker='o', markersize=2, linestyle='solid',
        #          label="linear_velocity(m/s)")
        # plt.plot(positions, rotation_rec, color='red', marker='*', markersize=2, linestyle='solid',
        #          label="angular_velocity(rad/s)")
        # plt.plot(positions, action_left_rec, color='yellow', marker='^', markersize=2, linestyle='solid',
        #          label="acc_l(m/s^2)")
        # plt.plot(positions, action_right_rec, color='purple', marker='d', markersize=2, linestyle='solid',
        #          label="acc_r(m/s^2)")
        # plt.legend(loc='upper left')
        print('finish')
        if args.traj:
            env.render('traj', args.video_file)
        else:
            if args.video_dir is not None:
                if policy_config.name == 'gcn':
                    args.video_file = os.path.join(args.video_dir,
                                                   policy_config.name + '_' + policy_config.gcn.similarity_function)
                else:
                    args.video_file = os.path.join(args.video_dir, policy_config.name)
                args.video_file = args.video_file + '_' + args.phase + '_' + str(args.test_case) + '.mp4'
            env.render('video', args.video_file)
        print('It takes %.2f seconds to finish. Final status is %s, cumulative_reward is %f'%(
                     env.global_time, info, cumulative_reward))
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        discomfort = 0
        min_dist = []
        cumulative_rewards = []
        average_returns = []
        returns_list = []
        collision_cases = []
        timeout_cases = []
        discomfort_nums = []
        k=env_config.env.test_size
        for i in range(k):
            states = []
            actions = []
            rewards = []
            dones = []
            num_discoms =[]
            done = False
            ob = env.reset(args.phase)
            last_pos = np.array(robot.get_position())
            while not done:
                num_discom = 0
                robot_pose, robot_velocity, robot_goal, obstacles = ob2req(ob, env.robot.get_full_state())
                rospy.wait_for_service('/test_mpc_crowd_sim_node/crowd_sim_info')
                try:
                    # 创建服务的处理句柄,可以像调用函数一样，调用句柄
                    teb_server = rospy.ServiceProxy('/test_mpc_crowd_sim_node/crowd_sim_info', TebCrowdSim)
                    resp1 = teb_server(robot_pose, robot_velocity, robot_goal, obstacles)
                    vx = resp1.velocity.linear.x
                    dt = resp1.velocity.linear.y
                    w = resp1.velocity.angular.z
                    if robot.kinematics is 'differential':
                        vel_left = (vx - w * env.robot.radius)
                        vel_right = (vx + w * env.robot.radius)
                        if dt == 0.0:
                            print("dt is 0")
                            dt = 0.5
                        al = (vel_left - env.robot.v_left) / (0.5 * dt)
                        ar = (vel_right - env.robot.v_right) / (0.5 * dt)
                        action = ActionDiff(al, ar)
                    elif robot.kinematics is 'unicycle':
                        theta = w * robot.time_step
                        action = ActionRot(vx, theta)
                    else:
                        action = ActionXY(vx * np.cos(robot.theta), vy * np.sin(robot.theta))
                    ob, reward, done, info = env.step(action)
                    # if phase in ['train', 'test']:
                    #     self.env.render(mode='debug')
                    # actually, final states of timeout cases is not terminal states
                    if isinstance(info, Timeout):
                        dones.append(False)
                    else:
                        dones.append(done)
                    rewards.append(reward)
                    if isinstance(info, Discomfort):
                        discomfort += 1
                        min_dist.append(info.min_dist)
                        num_discom = info.num
                    num_discoms.append(num_discom)

                    # 如果调用失败，可能会抛出rospy.ServiceException
                except rospy.ServiceException:
                    print("Service call failed:")
            # add the terminal state
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(env.global_time)
                if args.phase in ['test']:
                    print('collision happen %f and %d'%(env.global_time, i))
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                if args.phase in ['test']:
                    print('timeout happen and %f and %d'%(env.global_time, i))
                    rewards[-1] = rewards[-1]
                timeout_times.append(env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')
            discomfort_nums.append(sum(num_discoms))
            # cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
            #                                * reward for t, reward in enumerate(rewards)]))
            cumulative_rewards.append(sum(rewards))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(0.95, t * robot.time_step * robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            returns_list = returns_list + returns
            average_returns.append(np.average(returns))

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else env.time_limit
        avg_col_time = sum(collision_times) / len(collision_times) if collision_times else env.time_limit
        # extra_info = '' if episode is None else 'in episode {} '.format(episode)
        # extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} has success rate: {:.3f}, collision rate: {:.3f}, nav time: {:.3f}, col time: {:.3f}, total reward: {:.4f},'
                     ' average return: {:.4f}'. format(k, success_rate, collision_rate,
                                                       avg_nav_time, avg_col_time, sum(cumulative_rewards),
                                                       np.average(average_returns)))
        print('%d has success rate: %.4f, collision rate: %.4f, nav time: %.3f, col time: %.3f, total reward: %.4f,'
                     ' average return: %.4f'%(k, success_rate, collision_rate,
                                                       avg_nav_time, avg_col_time, sum(cumulative_rewards),
                                                       np.average(average_returns)))
        # if phase in ['val', 'test'] or imitation_learning:
        total_time = sum(success_times + collision_times + timeout_times) / robot.time_step
        logging.info('Frequency of being in danger: %.3f and average min separate distance in danger: %.2f',
                    discomfort / total_time, np.average(min_dist))
        print('Frequency of being in danger: %.3f and average min separate distance in danger: %.2f'%
              (discomfort / total_time, np.average(min_dist)))
        print('discomfor nums is %.0f and return is %.04f and length is %.0f'%( sum(discomfort_nums),
                     np.average(returns_list), len(returns_list)))
        if True:
            print('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            print('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))


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
    parser.add_argument('--human_num', type=int, default=11)
    parser.add_argument('--safety_space', type=float, default=0.2)
    parser.add_argument('--test_scenario', type=str, default=None)
    parser.add_argument('--plot_test_scenarios_hist', default=True, action='store_true')
    parser.add_argument('-d', '--planning_depth', type=int, default=None)
    parser.add_argument('-w', '--planning_width', type=int, default=None)
    parser.add_argument('--randomseed', type=int, default=7)
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