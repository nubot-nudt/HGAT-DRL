import os
import logging
import copy
import torch
import numpy as np
from tqdm import tqdm
from crowd_sim.envs.utils.info import *

class Explorer(object):
    def __init__(self, env, robot, device, writer, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.writer = writer
        self.memory = memory
        self.gamma = 0.95
        self.target_policy = target_policy
        self.statistics = None
        self.use_noisy_net = False

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None, epoch=None,
                       print_failure=False):
        self.robot.policy.set_phase(phase)
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
        if phase in ['test', 'val'] or imitation_learning:
            pbar = tqdm(total=k)
        else:
            pbar = None
        if self.robot.policy.name in ['model_predictive_rl', 'tree_search_rl']:
            if phase in ['test', 'val'] and self.use_noisy_net:
                self.robot.policy.model[2].eval()
            else:
                self.robot.policy.model[2].train()

        for i in range(k):
            ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            dones = []
            num_discoms =[]
            while not done:
                num_discom = 0
                action, action_index = self.robot.act(ob)
                ob, reward, done, info = self.env.step(action)
                states.append(self.robot.policy.last_state)
                # for TD3rl, append the velocity and theta
                actions.append(action_index)
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
            # add the terminal state
            states.append(self.robot.get_state(ob))
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
                if phase in ['test']:
                    print('collision happen %f', self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                if phase in ['test']:
                    print('timeout happen %f', self.env.global_time)
                    rewards[-1] = rewards[-1]
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                self.update_memory(states, actions, rewards, dones, imitation_learning)
            discomfort_nums.append(sum(num_discoms))
            # cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
            #                                * reward for t, reward in enumerate(rewards)]))
            cumulative_rewards.append(sum(rewards))
            returns = []
            for step in range(len(rewards)):
                step_return = sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                   * reward for t, reward in enumerate(rewards[step:])])
                returns.append(step_return)
            returns_list = returns_list + returns
            average_returns.append(average(returns))

            if pbar:
                pbar.update(1)

        success_rate = success / k
        collision_rate = collision / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit
        avg_col_time = sum(collision_times) / len(collision_times) if collision_times else self.env.time_limit
        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        extra_info = extra_info + '' if epoch is None else extra_info + ' in epoch {} '.format(epoch)
        logging.info('{:<5} {}has success rate: {:.3f}, collision rate: {:.3f}, nav time: {:.3f}, col time: {:.3f}, total reward: {:.4f},'
                     ' average return: {:.4f}'. format(phase.upper(), extra_info, success_rate, collision_rate,
                                                       avg_nav_time, avg_col_time, sum(cumulative_rewards),
                                                       average(average_returns)))
        # if phase in ['val', 'test'] or imitation_learning:
        total_time = sum(success_times + collision_times + timeout_times) / self.robot.time_step
        logging.info('Frequency of being in danger: %.3f and average min separate distance in danger: %.2f',
                    discomfort / total_time, average(min_dist))
        logging.info('discomfor nums is %.0f and return is %.04f and length is %.0f', sum(discomfort_nums),
                     average(returns_list), len(returns_list))
        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

        self.statistics = success_rate, collision_rate, avg_nav_time, avg_col_time, sum(cumulative_rewards), average(average_returns), discomfort, total_time

        return self.statistics

    def update_memory(self, states, actions, rewards, dones, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        
        for i, state in enumerate(states[:-1]):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                action = actions[i]
                done = torch.Tensor([dones[i]]).to(self.device)
                next_state = self.target_policy.transform(states[i+1])
                value = sum([pow(self.gamma, (t - i) * self.robot.time_step * self.robot.v_pref) * reward *
                             (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                next_state = states[i+1]
                action = actions[i]
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    value = 0
                value = torch.Tensor([value]).to(self.device)
                reward = torch.Tensor([rewards[i]]).to(self.device)
                done = torch.Tensor([dones[i]]).to(self.device)

            if self.target_policy.name == 'ModelPredictiveRL' or self.target_policy.name == 'TreeSearchRL':
                self.memory.push((state[0], state[1], action, value, done, reward, next_state[0], next_state[1]))
            elif self.target_policy.name == 'TD3RL':
                state = rotate_state2(state)
                next_state = rotate_state2(next_state)
                self.memory.push((state, action, value, done, reward, next_state))
            elif self.target_policy.name == 'RGCNRL':
                self.memory.push((state, action, value, done, reward, next_state))
            else:
                self.memory.push((state, value, done, reward, next_state))

    def log(self, tag_prefix, global_step):
        sr, cr, time, reward, avg_return, _, _, _ = self.statistics
        self.writer.add_scalar(tag_prefix + '/success_rate', sr, global_step)
        self.writer.add_scalar(tag_prefix + '/collision_rate', cr, global_step)
        self.writer.add_scalar(tag_prefix + '/time', time, global_step)
        self.writer.add_scalar(tag_prefix + '/reward', reward, global_step)
        self.writer.add_scalar(tag_prefix + '/avg_return', avg_return, global_step)


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0


def rotate_state(state):
    """
    Transform the coordinate to agent-centric.
    Input tuple include robot state tensor and human state tensor.
    robot state tensor is of size (batch_size, number, state_length)(for example 100*1*9)
    human state tensor is of size (batch_size, number, state_length)(for example 100*5*5)
    """
    # for robot
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
    #  0     1      2     3      4        5     6      7         8
    # for human
    #  'px', 'py', 'vx', 'vy', 'radius'
    #  0     1      2     3      4
    # for obstacle
    # 'px', 'py', 'radius'
    #  0     1     2
    # for wall
    # 'sx', 'sy', 'ex', 'ey'
    #  0     1     2     3
    assert len(state[0].shape) == 2
    if state[1] is None:
        robot_state = state[0]
        robot_feature_dim = state[0].shape[1]
        human_feature_dim = 5
        dx = robot_state[:, 5] - robot_state[:, 0]
        dy = robot_state[:, 6] - robot_state[:, 1]
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)
        radius_r = robot_state[:, 4].unsqueeze(1)
        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        rot = torch.atan2(dy, dx)
        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        vx = (robot_state[:, 2].unsqueeze(1) * cos_rot +
              robot_state[:, 3].unsqueeze(1) * sin_rot).reshape((1, -1))
        vy = (robot_state[:, 3].unsqueeze(1) * cos_rot -
              robot_state[:, 2].unsqueeze(1) * sin_rot).reshape((1, -1))
        v_pref = robot_state[:, 7].unsqueeze(1)
        theta = robot_state[:, 8].unsqueeze(1)
        px_r = torch.zeros_like(v_pref)
        py_r = torch.zeros_like(v_pref)
        new_robot_state = torch.cat((px_r, py_r, vx, vy, radius_r, dg, rot, v_pref, theta), dim=1)
        new_state = (new_robot_state, None)
        return new_state
    else:
        robot_state = state[0]
        human_state = state[1]
        obstacle_state = state[2]
        wall_state = state[3]
        human_num = human_state.shape[0]
        robot_num = robot_state.shape[0]
        obstacle_num = obstacle_state.shape[0]
        wall_num = wall_state.shape[1]
        dx = robot_state[:, 5] - robot_state[:, 0]
        dy = robot_state[:, 6] - robot_state[:, 1]
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)
        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        rot = torch.atan2(dy, dx)
        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        transform_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=0).reshape(2, 2)
        a = robot_state[:, 2:4]
        robot_velocities = torch.mm(robot_state[:, 2:4], transform_matrix)
        radius_r = robot_state[:, 4].unsqueeze(1)
        v_pref = robot_state[:, 7].unsqueeze(1)
        target_heading = torch.zeros_like(radius_r)
        pos_r = torch.zeros_like(robot_velocities)
        cur_heading = (robot_state[:, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
        new_robot_state = torch.cat((robot_velocities, dg, v_pref, cur_heading),
                                    dim=1)

        human_positions = human_state[:, 0:2] - robot_state[:, 0:2]
        human_positions = torch.mm(human_positions, transform_matrix)
        human_velocities = human_state[:, 2:4]
        human_velocities = torch.mm(human_velocities, transform_matrix)
        human_radius = human_state[:, 4].unsqueeze(1) + 0.3
        new_human_state = torch.cat((human_positions, human_velocities, human_radius), dim=1)

        wall_start_positions = wall_state[:, 0:2] - robot_state[:, 0:2]
        wall_start_positions = torch.mm(wall_start_positions, transform_matrix)
        wall_end_positions = wall_state[:, 2:4] - robot_state[:, 0:2]
        wall_end_positions = torch.mm(wall_end_positions, transform_matrix)
        wall_radius = torch.zeros((wall_state.shape[0], 1)) + 0.3
        new_wall_states = torch.cat((wall_start_positions, wall_end_positions, wall_radius), dim=1)
        if len(obstacle_state.shape) == 2:
            obstacle_positions = obstacle_state[:, 0:2] - robot_state[:, 0:2]
            obstacle_positions = torch.mm(obstacle_positions, transform_matrix)
            obstacle_radius = obstacle_state[:, 2].unsqueeze(1) + 0.3
            new_obstacle_states = torch.cat((obstacle_positions, obstacle_radius), dim=1)
            robot_feature_dim = new_robot_state.shape[1]
            human_feature_dim = new_human_state.shape[1]
            obstacle_feature_dim = new_obstacle_states.shape[1]
            wall_feature_dim = new_wall_states.shape[1]
            robot_zero_feature = torch.zeros([robot_num, human_feature_dim + obstacle_feature_dim + wall_feature_dim])
            human_zero_feature1 = torch.zeros([human_num, robot_feature_dim])
            human_zero_feature2 = torch.zeros([human_num, obstacle_feature_dim + wall_feature_dim])
            obstacle_zero_feature1 = torch.zeros([obstacle_num, robot_feature_dim + human_feature_dim])
            obstacle_zero_feature2 = torch.zeros([obstacle_num, wall_feature_dim])
            wall_zero_feature = torch.zeros([wall_num, robot_feature_dim + human_feature_dim + obstacle_feature_dim])
            new_robot_state = torch.cat((new_robot_state, robot_zero_feature), dim=1)
            new_human_state = torch.cat((human_zero_feature1, new_human_state, human_zero_feature2), dim=1)
            new_obstacle_states = torch.cat((obstacle_zero_feature1, new_obstacle_states, obstacle_zero_feature2),
                                            dim=1)
            new_wall_states = torch.cat((wall_zero_feature, new_wall_states), dim=1)

            new_state = torch.cat((new_robot_state, new_human_state, new_obstacle_states, new_wall_states), dim=0)
        else:
            robot_feature_dim = new_robot_state.shape[1]
            human_feature_dim = new_human_state.shape[1]
            obstacle_feature_dim = 3
            wall_feature_dim = new_wall_states.shape[1]
            robot_zero_feature = torch.zeros([robot_num, human_feature_dim + obstacle_feature_dim + wall_feature_dim])
            human_zero_feature1 = torch.zeros([human_num, robot_feature_dim])
            human_zero_feature2 = torch.zeros([human_num, obstacle_feature_dim + wall_feature_dim])
            obstacle_zero_feature1 = torch.zeros([obstacle_num, robot_feature_dim + human_feature_dim])
            obstacle_zero_feature2 = torch.zeros([obstacle_num, wall_feature_dim])
            wall_zero_feature = torch.zeros([wall_num, robot_feature_dim + human_feature_dim + obstacle_feature_dim])
            new_robot_state = torch.cat((new_robot_state, robot_zero_feature), dim=1)
            new_human_state = torch.cat((human_zero_feature1, new_human_state, human_zero_feature2), dim=1)
            new_wall_states = torch.cat((wall_zero_feature, new_wall_states), dim=1)
            new_state = torch.cat((new_robot_state, new_human_state, new_wall_states), dim=0)
        return new_state


def rotate_state2(state):
    """
                Transform the coordinate to agent-centric.
                Input tuple include robot state tensor and human state tensor.
                robot state tensor is of size (batch_size, number, state_length)(for example 100*1*9)
                human state tensor is of size (batch_size, number, state_length)(for example 100*5*5)
                """
    # for robot
    # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
    #  0     1      2     3      4        5     6      7         8
    # for human
    #  'px', 'py', 'vx', 'vy', 'radius'
    #  0     1      2     3      4
    # for obstacle
    # 'px', 'py', 'radius'
    #  0     1     2
    # for wall
    # 'sx', 'sy', 'ex', 'ey'
    #  0     1     2     3
    assert len(state[0].shape) == 2
    robot_state = state[0]
    human_state = state[1]
    obstacle_state = state[2]
    wall_state = state[3]
    human_num = human_state.shape[0]
    robot_num = robot_state.shape[0]
    obstacle_num = obstacle_state.shape[0]
    wall_num = wall_state.shape[0]

    dx = robot_state[:, 5] - robot_state[:, 0]
    dy = robot_state[:, 6] - robot_state[:, 1]
    dx = dx.unsqueeze(1)
    dy = dy.unsqueeze(1)
    dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
    rot = torch.atan2(dy, dx)
    cos_rot = torch.cos(rot)
    sin_rot = torch.sin(rot)
    transform_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=0).reshape(2, 2)
    a = robot_state[:, 2:4]
    robot_velocities = torch.mm(robot_state[:, 2:4], transform_matrix)
    radius_r = robot_state[:, 4].unsqueeze(1)
    v_pref = robot_state[:, 7].unsqueeze(1)
    target_heading = torch.zeros_like(radius_r)
    pos_r = torch.zeros_like(robot_velocities)
    cur_heading = (robot_state[:, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
    new_robot_state = torch.cat((robot_velocities, dg, v_pref, cur_heading),
                                dim=1)

    human_positions = human_state[:, 0:2] - robot_state[:, 0:2]
    human_positions = torch.mm(human_positions, transform_matrix)
    human_velocities = human_state[:, 2:4]
    human_velocities = torch.mm(human_velocities, transform_matrix)
    human_radius = human_state[:, 4].unsqueeze(1) + 0.3
    new_human_state = torch.cat((human_positions, human_velocities, human_radius), dim=1)

    if len(obstacle_state.shape) == 2:
        obstacle_positions = obstacle_state[:, 0:2] - robot_state[:, 0:2]
        obstacle_positions = torch.mm(obstacle_positions, transform_matrix)
        obstacle_radius = obstacle_state[:, 2].unsqueeze(1) + 0.3
        obstacle_velocity = torch.zeros_like(obstacle_positions)
        obs_human = torch.cat((obstacle_positions, obstacle_velocity, obstacle_radius), dim=1)
        new_human_state = torch.cat((new_human_state, obs_human), dim=0)
        new_obstacle_states = torch.cat((obstacle_positions, obstacle_radius), dim=1)
        robot_feature_dim = new_robot_state.shape[1]
        human_feature_dim = new_human_state.shape[1]
        obstacle_feature_dim = new_obstacle_states.shape[1]
        wall_feature_dim = 5
        robot_zero_feature = torch.zeros(
            [robot_num, human_feature_dim + obstacle_feature_dim + wall_feature_dim])
        human_zero_feature1 = torch.zeros([human_num + obstacle_num, robot_feature_dim])
        human_zero_feature2 = torch.zeros([human_num + obstacle_num, obstacle_feature_dim + wall_feature_dim])
        new_robot_state = torch.cat((new_robot_state, robot_zero_feature), dim=1)
        new_human_state = torch.cat((human_zero_feature1, new_human_state, human_zero_feature2), dim=1)
        new_state = torch.cat((new_robot_state, new_human_state),
                              dim=0)
    return new_state