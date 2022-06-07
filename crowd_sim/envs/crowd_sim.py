import logging
import random
import math
import copy
import gym
from gym import spaces
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm
from crowd_nav.utils.rvo_inter import rvo_inter
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.state import tensor_to_joint_state, JointState
from crowd_sim.envs.utils.action import ActionRot, ActionDiff
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.obstacle import Obstacle
from crowd_sim.envs.utils.wall import Wall
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import point_to_segment_dist, counterclockwise, point_in_poly, theta_mod

class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.obstacles = None
        self.walls = None
        self.global_time = None
        self.robot_sensor_range = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.goal_factor = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        self.nonstop_human = None
        self.centralized_planning = None
        self.centralized_planner = None
        self.test_changing_size = False

        self.static_obstacle_num = 3
        self.wall_num = 4

        self.fig = None
        self.ax = None
        self.poly_obstacles = []

        # for visualization
        self.states = None
        self.action_values = None
        self.attention_weights = None
        self.robot_actions = None
        self.rewards = None
        self.As = None
        self.Xs = None
        self.feats = None
        self.trajs = list()
        self.panel_width = 10
        self.panel_height = 10
        self.panel_scale = 1
        self.random_seed = 0
        self.test_scene_seeds = []
        self.dynamic_human_num = []
        self.human_starts = []
        self.human_goals = []
        self.phase_num = 0
        self.last_state = None
        self.rvo_inter = rvo_inter(neighbor_region=6, neighbor_num=20, vxmax=1, vymax=1, acceler=1.0,
                                   env_train=True,
                                   exp_radius=0.0, ctime_threshold=3.0, ctime_line_threshold=2.0)
        # 动作空间: 速度，朝向
        self.action_space = spaces.Box(
            low=np.array([0, -np.pi]),
            high=np.array([1, np.pi]),
            dtype=np.float32
        )
        self.phase = None

    def configure(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range
        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.goal_factor = config.reward.goal_factor
        self.re_rvo = config.reward.re_rvo
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.discomfort_dist = config.reward.discomfort_dist
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size,
                          'test': config.env.test_size}
        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius
        self.human_num = config.sim.human_num
        self.human_num = 0
        self.static_obstacle_num = 3
        self.wall_num = 0

        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")
        logging.info('Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))
        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_phase(self, phase_num):
        self.phase_num = phase_num
        if self.phase_num == 0:
            self.static_obstacle_num = 3
            self.wall_num = 2
            self.human_num = 1
        elif self.phase_num == 1:
            self.static_obstacle_num = 3
            self.wall_num = 2
            self.human_num = 3
        elif self.phase_num == 2:
            self.static_obstacle_num = 3
            self.wall_num = 2
            self.human_num = 5
        elif self.phase_num == 3:
            self.static_obstacle_num = 3
            self.wall_num = 2
            self.human_num = 5
        elif self.phase_num == 10: #for test
            self.static_obstacle_num = 3
            self.wall_num = 0
            self.human_num = 5

    def set_robot(self, robot):
        self.robot = robot

        if self.robot.kinematics == "holonomic":
            # 动作空间: 速度，朝向
            self.action_space = spaces.Box(
                low=np.array([0, -np.pi]),
                high=np.array([1, np.pi]),
                dtype=np.float32
            )
        elif self.robot.kinematics =='unicycle':
            self.action_space = spaces.Box(
                low=np.array([0, -self.robot.rotation_constraint]),
                high=np.array([1, self.robot.rotation_constraint]),
                dtype=np.float32
            )
        elif self.robot.kinematics =='differential':
            self.action_space = spaces.Box(
                low=np.array([-1.0,-1.0]),
                high=np.array([1.0,1.0]),
                dtype=np.float32
            )
        logging.info('rotation constraint: {}'.format(self.robot.rotation_constraint))

    def set_human_num(self, human_num):
        self.human_num = human_num
        logging.info('human number is {}'.format(self.human_num))

    def generate_human(self, human=None, non_stop=False, square=False):
        if human is None:
            human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if square is False and non_stop is False:
            for sample_count in range(200):
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * human.v_pref
                py_noise = (np.random.random() - 0.5) * human.v_pref
                ex_noise = (np.random.random() - 0.5) * human.v_pref
                ey_noise = (np.random.random() - 0.5) * human.v_pref
                px = (self.circle_radius + 0.5) * np.cos(angle) + px_noise
                py = (self.circle_radius + 0.5) * np.sin(angle) + py_noise
                gx = -(self.circle_radius + 0.5) * np.cos(angle) + ex_noise
                gy = -(self.circle_radius + 0.5) * np.sin(angle) + ey_noise
                collide = False
                for agent in [self.robot]:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                for agent in self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((px - agent.px, py - agent.py)) < min_dist or \
                            norm((px - agent.gx, py - agent.gy)) < min_dist:
                        collide = True
                        break
                for wall in self.walls:
                    if point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, px, py) < human.radius + 0.3 and \
                            point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx, gy) < human.radius + 0.3:
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(px, py, poly_obs) or point_in_poly(gx, gy, poly_obs):
                        collide = True
                        break
                if not collide:
                    break
            # px = 0.0
            # py = 0.0
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)
        elif square is False and non_stop is True:
            for sample_count in range(200):
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px = human.px
                py = human.py
                gx_noise = (np.random.random() - 0.5) * human.v_pref
                gy_noise = (np.random.random() - 0.5) * human.v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                for agent in [self.robot]:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                for wall in self.walls:
                    if point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx, gy) < human.radius + self.robot.radius:
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(gx, gy, poly_obs):
                        collide = True
                        break
                if not collide:
                    break
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)

        elif square is True and non_stop is False:
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            for sample_count in range(200):
                px = np.random.random() * self.square_width * 0.5 * sign
                py = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        break
                for wall in self.walls:
                    if point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, px, py) < human.radius + self.robot.radius:
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(px, py, poly_obs):
                        collide = True
                        break
                if not collide:
                    break
            for sample_count in range(200):
                gx = np.random.random() * self.square_width * 0.5 * (- sign)
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if  norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                for wall in self.walls:
                    if point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx, gy) < human.radius + self.robot.radius:
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(gx, gy, poly_obs) or point_in_poly(gx, gy, poly_obs):
                        collide = True
                        break
                if not collide:
                    break
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)
        elif square is True and non_stop is True:
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            goal_count = 0
            for sample_count in range(200):
                goal_count = goal_count + 1
                px = human.px
                py = human.py
                gx = np.random.random() * self.square_width * 0.5 * (- sign)
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                for wall in self.walls:
                    if point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx,
                                             gy) < human.radius + self.robot.radius:
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(gx, gy, poly_obs) or point_in_poly(gx, gy, poly_obs):
                        collide = True
                        break
                if not collide:
                    break
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_static_obstacle(self, obstacle=None):
        self.obstacles = []
        for i in range(self.static_obstacle_num):
            obstacle = Obstacle()
        # if self.randomize_attributes:
            obstacle.sample_random_attributes()
        # else:
        #     obstacle.radius = 0.3
            sample_count = 0
            for sample_count in range(200):
                px = (np.random.random() - 0.5) * self.square_width * 0.8
                py = (np.random.random() - 0.5) * self.circle_radius * 2
                obstacle.set(px, py, obstacle.radius)
                collide = False
                for agent in [self.robot] + self.humans:
                    if norm((px - agent.px, py - agent.py)) < obstacle.radius + agent.radius + 0.5 or \
                            norm((px - agent.gx, py - agent.gy)) < obstacle.radius + agent.radius + 0.5:
                        collide = True
                        break
                for agent in self.obstacles:
                    if norm((px - agent.px, py - agent.py)) < obstacle.radius + agent.radius + 0.5:
                        collide = True
                        break
                for wall in self.walls:
                    if point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, px, py) < obstacle.radius + 0.8:
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(px, py, poly_obs):
                        collide = True
                        break
                if not collide:
                    break
            if sample_count < 200:
                self.obstacles.append(obstacle)

    def generate_center_obstacle(self, obstacle=None):
        corridor_width = self.square_width - 1.0
        transfer_width = 3.0
        center_x = (np.random.random() - 0.5) * 2
        center_y = (np.random.random() - 0.5) * 2
        width = np.clip(np.random.normal(2, 1.0), 1, 3)
        length = np.clip(np.random.normal(2, 1.0), 1, 3)
        x1 = center_x - width / 2.0
        x2 = center_x + width / 2.0
        y1 = center_y - length / 2.0
        y2 = center_y + length / 2.0
        if self.phase_num == 10:
            # y1 = -0.5
            # y2 = 1.5
            # x1 = -0.5
            # x2 = 0.5

            y1 = -0.5
            y2 = 0.5
            x1 = -1.25
            x2 = 1.25

        transfer_vertex =([x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1,y1])
        # transfer_vertex = ([corridor_width/6, -transfer_width / 2], [corridor_width/2, -transfer_width / 2],
        # [corridor_width/2, transfer_width / 2], [corridor_width/6, transfer_width / 2], [corridor_width/6, -transfer_width / 2])
        for i in range(len(transfer_vertex)-1):
            self.walls.append(self.generate_wall(transfer_vertex[i], transfer_vertex[i+1]))
        self.poly_obstacles.clear()
        self.poly_obstacles.append(transfer_vertex)


    def generate_airport_transfer(self):
        self.generate_corridor_scenario()
        self.generate_center_obstacle()
        # for i in range(self.wall_num):
        #     self.walls.append(self.generate_line_obstacle())
        # self.generate_transfer()
        # self.generate_open_scenario()

    def generate_constrained_room(self):
        self.generate_corridor_scenario()
        if self.phase_num > 0:
            for i in range(self.wall_num):
                self.walls.append(self.generate_transfer())

    def generate_doorway_scenario(self):
        room_width = self.square_width - 1
        room_length = self.square_width - 1
        wall_vertex = (
            [-room_width / 2, -room_length / 2], [room_width / 2, -room_length / 2], [room_width / 2, room_length / 2],
            [-room_width / 2, room_length / 2], [-room_width / 2, -room_length / 2])
        for i in range(len(wall_vertex) - 1):
            self.walls.append(self.generate_wall(wall_vertex[i], wall_vertex[i + 1]))

    def generate_corridor_scenario(self):
        corridor_width = self.square_width
        corridor_length = self.square_width * 2.0
        self.walls.append(self.generate_wall([-corridor_width / 2, -corridor_length / 2], [-corridor_width / 2, corridor_length / 2]))
        self.walls.append(self.generate_wall([corridor_width / 2, -corridor_length / 2], [corridor_width / 2, corridor_length / 2]))

    def generate_open_scenario(self):
        room_width = self.square_width - 1
        room_length = self.square_width - 1
        wall_vertex = (
        [-room_width / 2, -room_length / 2], [room_width / 2, -room_length / 2], [room_width / 2, room_length / 2],
        [-room_width / 2, room_length / 2], [-room_width / 2, -room_length / 2])
        for i in range(len(wall_vertex)-1):
            self.walls.append(self.generate_wall(wall_vertex[i], wall_vertex[i+1]))

    def generate_transfer(self):
        corridor_width = self.square_width - 1.0
        transfer_width = 2.0
        x1 = 0 - 2 + 1
        x2 = 2 - 1
        y1 = -transfer_width/2
        y2 = transfer_width/2.0
        transfer_vertex =([x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1,y1])
        # transfer_vertex = ([corridor_width/6, -transfer_width / 2], [corridor_width/2, -transfer_width / 2],
        # [corridor_width/2, transfer_width / 2], [corridor_width/6, transfer_width / 2], [corridor_width/6, -transfer_width / 2])
        for i in range(len(transfer_vertex)-1):
            if i ==0:
                self.walls.append(self.generate_wall(transfer_vertex[i], transfer_vertex[i+1]))
        self.poly_obstacles.clear()
        self.poly_obstacles.append(transfer_vertex)

    def generate_ward_scenario(self):
        room_width = self.square_width - 1
        room_length = self.square_width - 1
        self.walls = []
        wall_vertex = (
            [-room_width / 2, -room_length / 2], [room_width / 2, -room_length / 2], [room_width / 2, room_length / 2],
            [-room_width / 2, room_length / 2], [-room_width / 2, -room_length / 2])
        for i in range(len(wall_vertex) - 1):
            self.walls.append(self.generate_wall(wall_vertex[i], wall_vertex[i + 1]))

    def generate_wall(self, start_position, end_position, wall=None):
        wall = Wall(self.config)
        wall.set_position(start_position, end_position)
        return wall

    def generate_line_obstacle(self, wall=None):
        wall = Wall(self.config)
        while True:
            start_x = (np.random.random() - 0.5) * self.square_width * 0.6
            start_y = (np.random.random() - 0.5) * self.square_width * 0.6
            mean_length = self.circle_radius * 0.75
            wall_length = np.random.normal(mean_length, 0.1)
            theta = (np.random.random() - 0.5) * np.pi / 9 + len(self.walls) * np.pi / 2
            if wall_length == 0.0:
                print('error')
                break
            else:
                end_y = start_y + np.sin(theta) * wall_length
                end_x = start_x + np.cos(theta) * wall_length
                end_y = np.clip(end_y, -0.5*self.square_width, 0.5*self.square_width)
                end_x = np.clip(end_x, -0.5*self.square_width, 0.5*self.square_width)
            collide = False
            if not collide:
                wall.set_position([start_x, start_y], [end_x, end_y])
                break
        return wall



    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        train_seed_begin = [0, 10, 100, 1000, 10000]
        val_seed_begin = [0, 10, 100, 1000, 10000]
        test_seed_begin = [0, 10, 100, 1000, 10000]
        base_seed = {'train': self.case_capacity['val'] + self.case_capacity['test'] + train_seed_begin[1],
                     'val': 0 + val_seed_begin[1], 'test': self.case_capacity['val']+test_seed_begin[2]+1000}
        robot_theta = np.pi / 2 + np.random.random() * np.pi / 4.0 - np.pi / 8.0
        if self.phase_num == 10:
            target_x = 0
            target_y = self.circle_radius
            robot_theta = 0
        else:
            target_x = (np.random.random() - 0.5) * self.square_width * 0.8
            target_y = self.circle_radius
        target_x = 0
        target_y = self.circle_radius
        # robot_theta = 0
        # target_x = 0
        # target_y = self.circle_radius
        self.robot.set(0, -self.circle_radius, target_x, target_y, 0, 0, robot_theta)
        self.random_seed = base_seed[phase] + self.case_counter[phase]
        np.random.seed(self.random_seed)
        if self.case_counter[phase] >= 0:
            # self.generate_constrained_room()
            self.walls = []
            self.generate_airport_transfer()
            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
                # print('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))
            if not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                # human_num = 1
                self.current_scenario = 'circle_crossing'
            else:
                self.current_scenario = self.test_scenario
                # human_num = self.human_num
            self.humans = []
            for i in range(self.human_num):
                if self.current_scenario == 'circle_crossing' and i < 5:
                    self.humans.append(self.generate_human())
                else:
                    self.humans.append(self.generate_human(square=True))
            self.generate_static_obstacle()
            # self.generate_center_obstcale()
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError
        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.time_step = self.time_step

        self.states = list()
        self.robot_actions = list()
        self.rewards = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()
        if hasattr(self.robot.policy, 'get_matrix_A'):
            self.As = list()
        if hasattr(self.robot.policy, 'get_feat'):
            self.feats = list()
        if hasattr(self.robot.policy, 'get_X'):
            self.Xs = list()
        if hasattr(self.robot.policy, 'traj'):
            self.trajs = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob_human = self.compute_observation_for(self.robot)
            if self.obstacles is not None:
                ob_obstacles = [obstacle.get_observable_state() for i, obstacle in enumerate(self.obstacles)]
            else:
                ob_obstacles = None
            if self.walls is not None:
                ob_walls = [wall.get_observable_state() for i, wall in enumerate(self.walls)]
            else:
                ob_walls = None
            ob = (ob_human, ob_obstacles, ob_walls)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError
        return ob

    def onestep_lookahead(self, action):

        return self.step(action, update=False)

    def reward_cal(self, action, human_actions):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        """Firstly, predict the robot position"""
        pre_robot_pos_x, pre_robot_pos_y = self.robot.compute_position(action, self.time_step)
        weight_goal = self.goal_factor
        weight_safe = self.discomfort_penalty_factor
        weight_terminal = 1.0
        re_collision = self.collision_penalty
        re_arrival = self.success_reward
        # collision detection
        dmin = float('inf')
        collision = False
        safety_penalty = 0.0
        num_discom = 0
        """Secondly, deal with humans """
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            end_human_x = human_actions[i].vx * self.time_step + human.px
            end_human_y = human_actions[i].vy * self.time_step + human.py
            ex = end_human_x - pre_robot_pos_x
            ey = end_human_y - pre_robot_pos_y
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.debug("Collision: distance between robot and pedestrian{} is {:.2E} at time {:.2E}".format(
                    human.id, closest_dist, self.global_time))
            if closest_dist < dmin:
                dmin = closest_dist
            if closest_dist < self.discomfort_dist:
                safety_penalty = safety_penalty + (closest_dist - self.discomfort_dist)
                num_discom = num_discom + 1

        """ Thirdly, deal with obstacles"""
        for i, obstacle in enumerate(self.obstacles):
            px = obstacle.px - self.robot.px
            py = obstacle.py - self.robot.py
            ex = obstacle.px - pre_robot_pos_x
            ey = obstacle.py - pre_robot_pos_y
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - obstacle.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.debug("Collision: distance between robot and obstacle{} is {:.2E} at time {:.2E}".format(
                    i, closest_dist, self.global_time))
                num_discom = num_discom + 1
            # if closest_dist < dmin:
            # dmin = closest_dist
            if closest_dist < self.discomfort_dist * 0.5:
                safety_penalty = safety_penalty + (closest_dist - self.discomfort_dist * 0.5) * 0.5
                num_discom = num_discom + 1

        """ Then, deal with walls"""
        for i, wall in enumerate(self.walls):
            # across the wall #
            if (counterclockwise(wall.sx, wall.sy, wall.ex, wall.ey, self.robot.px, self.robot.py) !=
                counterclockwise(wall.sx, wall.sy, wall.ex, wall.ey, pre_robot_pos_x, pre_robot_pos_y)) and \
                    (counterclockwise(self.robot.px, self.robot.py, pre_robot_pos_x, pre_robot_pos_y, wall.sx, wall.sy) !=
                     counterclockwise(self.robot.px, self.robot.py, pre_robot_pos_x, pre_robot_pos_y, wall.ex, wall.ey)):
                closest_dist = 0.0
            else:
                min_dis_start = point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, self.robot.px, self.robot.py)
                min_dis_end = point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, pre_robot_pos_x, pre_robot_pos_y)
                if min_dis_end < min_dis_start:
                    closest_dist = min_dis_end
                else:
                    closest_dist = min_dis_start
            closest_dist = closest_dist - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.debug("Collision: distance between robot and wall {} is {:.2E} at time {:.2E}".format(
                    i, closest_dist, self.global_time))
                num_discom = num_discom + 1
            # if closest_dist < dmin:
            #     dmin = closest_dist
            if closest_dist < self.discomfort_dist * 0.5:
                safety_penalty = safety_penalty + (closest_dist - self.discomfort_dist * 0.5) * 0.5
                num_discom = num_discom + 1
        # collision detection between humans
        # human_num = len(self.humans)
        # for i in range(human_num):
        #     for j in range(i + 1, human_num):
        #         dx = self.humans[i].px - self.humans[j].px
        #         dy = self.humans[i].py - self.humans[j].py
        #         dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
        #         if dist < 0:
        #             # detect collision but don't take humans' collision into account
        #             logging.debug('Collision happens between humans in step()')

        """check if reaching the goal"""
        end_position = np.array([pre_robot_pos_x, pre_robot_pos_y])
        cur_position = np.array((self.robot.px, self.robot.py))
        goal_position = np.array(self.robot.get_goal_position())
        reward_goal = (norm(cur_position - goal_position) - norm(end_position - goal_position))
        reaching_goal = norm(end_position - goal_position) < self.robot.radius

        reward_col = 0.0
        reward_arrival = 0.0
        if self.global_time >= self.time_limit - 1:
            done = True
            info = Timeout()
        elif collision:
            reward_col = re_collision
            done = True
            info = Collision()
        elif reaching_goal:
            reward_arrival = re_arrival
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            done = False
            info = Discomfort(dmin)
            info.num = num_discom
        else:
            done = False
            info = Nothing()
        reward_terminal = reward_arrival + reward_col
        reward = weight_terminal * reward_terminal + weight_goal * reward_goal + weight_safe * safety_penalty
        return reward, done, info

    def rvo_reward_cal(self, ob, reward_parameter=(0.2, 0.1, 0.1, 0.2, 0.2, 1, -10, 20)):
        robot_state, human_state, obstacle_state, wall_state = self.robot.get_state(ob)
        robot_state_array = robot_state.numpy()
        human_state_array = human_state.numpy()
        obstacle_state_array = obstacle_state.numpy()
        wall_state_array = wall_state.numpy()
        vo_flag, min_exp_time, min_dis = self.rvo_inter.config_vo_reward(robot_state_array, human_state_array,
                                                                   obstacle_state_array, wall_state_array)
        p1, p2, p3, p4, p5, p6, p7, p8 = reward_parameter
        if min_exp_time < 0:
            min_exp_time = 0
        exp_time_reward = - 0.5 / (min_exp_time + 0.5)
        # rvo reward
        p1 = 0.2
        p4 = 0.4
        p5 = 0
        rvo_reward = 0.0
        if vo_flag:
            rvo_reward = -0.1 + p1 * exp_time_reward  # -0.1 to -0.2
            if min_exp_time < 0.5:
                rvo_reward = -0.1 + p4 * exp_time_reward  # -0.3 to -0.5
        else:
            rvo_reward = p5
        # rvo_reward = np.round(rvo_reward, 4)
        # if rvo_reward > 0.0:
        #     print("error rvo reward")
        return rvo_reward

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            self.centralized_planner.set_walls(self.walls)
            self.centralized_planner.set_static_obstacles(self.obstacles)
            if self.robot.visible:
                if self.robot.kinematics == 'differential':
                    robot_state = self.robot.get_full_state()
                    linear_vel = 0.5 * (robot_state.vx + robot_state.vy)
                    robot_state.vx = linear_vel * np.cos(robot_state.theta)
                    robot_state.vy = linear_vel * np.sin(robot_state.theta)
                    agent_states.append(robot_state)
                else:
                    agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states)
        else:
            human_actions = []
            for human in self.humans:
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob))
        reward, done, info = self.reward_cal(action, human_actions)
        if update:
            # store state, action value and attention weights
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(self.robot.policy.get_attention_weights())
            if hasattr(self.robot.policy, 'get_matrix_A'):
                self.As.append(self.robot.policy.get_matrix_A())
            if hasattr(self.robot.policy, 'get_feat'):
                self.feats.append(self.robot.policy.get_feat())
            if hasattr(self.robot.policy, 'get_X'):
                self.Xs.append(self.robot.policy.get_X())
            if hasattr(self.robot.policy, 'traj'):
                self.trajs.append(self.robot.policy.get_traj())

            # update all agents
            self.robot.step(action)
            for human, action in zip(self.humans, human_actions):
                human.step(action)
                if self.nonstop_human and human.reached_destination():
                    human.reach_count = human.reach_count + 1
                    if human.reach_count > 2 and norm((human.px - self.robot.px, human.py - self.robot.py)) > human.radius + self.robot.radius + 0.5:
                        if self.current_scenario == 'circle_crossing':
                            self.generate_human(human, non_stop=True)
                            human.reach_count = 0
                        else:
                            self.generate_human(human, non_stop=True, square=True)
                            human.reach_count = 0

            self.global_time += self.time_step
            self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans],
                                [human.id for human in self.humans]])
            self.robot_actions.append(action)
            self.rewards.append(reward)

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob_human = [human.get_observable_state() for human in self.humans]
                ob_obstacles = [obstacle.get_observable_state() for i, obstacle in enumerate(self.obstacles)]
                ob_walls = [wall.get_observable_state() for i, wall in enumerate(self.walls)]
                ob = (ob_human, ob_obstacles, ob_walls)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob_human = [human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)]
                ob_obstacles = [obstacle.get_observable_state() for i, obstacle in enumerate(self.obstacles)]
                ob_walls = [wall.get_observable_state() for i, wall in enumerate(self.walls)]
                ob = (ob_human, ob_obstacles, ob_walls)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        rvo_reward = self.rvo_reward_cal(ob)
        reward = reward + self.re_rvo * rvo_reward
        # if info ==Collision():
        #     reward = rvo_reward - 15
        # elif info ==ReachGoal():
        #     reward = rvo_reward + 20
        # else:
        #     reward = rvo_reward
        return ob, reward, done, info

    def peds_predict(self, agent_states, robot_state):
        if self.robot.visible:
            agent_states.append(robot_state)
            human_actions = self.centralized_planner.predict(agent_states)[:-1]
        else:
            human_actions = self.centralized_planner.predict(agent_states)
        return human_actions

    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                if self.test_changing_size is False:
                    ob.append(human.get_observable_state())
                else:
                    dis2 = (human.px - agent.px) * (human.px - agent.px) + (human.py - agent.py) * (human.py - agent.py)
                    if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                        ob.append(human.get_observable_state())
        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def render(self, mode='video', output_file=None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.3
        y_offset = 0.4
        cmap = plt.cm.get_cmap('terrain', 200)
        cmap2 = plt.cm.get_cmap('tab10', 10)
        robot_color = 'black'
        arrow_style = patches.ArrowStyle.Fancy(head_length=4, head_width=2, tail_width=.6)

            # patches.ArrowStyle("->", head_length=4, head_width=2)
        display_numbers = True

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.panel_width/2 -1 , self.panel_width/2 + 1)
            ax.set_ylim(-self.panel_height/2-0.5, self.panel_height/2+0.5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap2(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                           color=human_colors[i],
                                           marker='*', linestyle='None', markersize=10)
                ax.add_artist(human_goal)
                for j in range(len(human.start_pos)):
                    pos = human.start_pos[j]
                    if j ==0:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker=4,
                                                   linestyle='None', markersize=10)
                    else:
                        test_start = mlines.Line2D([pos[0]], [pos[1]], color=human_colors[i], marker='s',
                                                   linestyle='None', markersize=10)
                    ax.add_artist(test_start)
                # human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                #                             color=human_colors[i],
                #                             marker='s', linestyle='None', markersize=10)
                # ax.add_artist(human_start)
            for i in range(len(self.obstacles)):
                obstacle = self.obstacles[i]
                obstacle_mark = plt.Circle(obstacle.get_position(), obstacle.radius, fill=True, color='grey')
                ax.add_artist(obstacle_mark)

            for i in range(len(self.walls)):
                wall = self.walls[i]
                wall_line = mlines.Line2D([wall.sx, wall.ex], [wall.sy, wall.ey], color='black', marker='.',linestyle='solid', markersize=5)
                ax.add_artist(wall_line)
            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [[self.states[i][1][j].position for j in range(len(self.humans))]
                               for i in range(len(self.states))]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color='black', marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            ax.add_artist(goal)
            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius/4, fill=True, color=robot_color)
                    humans = [plt.Circle(human_positions[k][i], self.humans[i].radius/4, fill=True, color=human_colors[i])
                              for i in range(len(self.humans))]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 5 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [plt.text(agents[i].center[0]+0.1, agents[i].center[1]+0.1,
                                      '{:.1f}'.format(global_time),
                                      color='black', fontsize=12) for i in range(self.human_num + 1)]
                    for time in times:
                       ax.add_artist(time)
                if k != 0:
                    nav_direction = plt.arrow(self.states[k - 1][0].px, self.states[k - 1][0].py,
                                              self.states[k][0].px - self.states[k - 1][0].px,
                                              self.states[k][0].py - self.states[k - 1][0].py,
                                            length_includes_head=True, head_width=0.08, lw=0.8, color='black')
                    human_directions = [plt.arrow(self.states[k - 1][1][i].px, self.states[k - 1][1][i].py, self.states[k][1][i].px - self.states[k - 1][1][i].px,self.states[k][1][i].py - self.states[k - 1][1][i].py,
                              length_includes_head=True, head_width=0.08, lw=0.5,
                              color=human_colors[i]) for i in range(self.human_num)]
                    #
                    # human_directions = [plt.Line2D((self.states[k - 1][1][i].px, self.states[k][1][i].px),
                    #                                (self.states[k - 1][1][i].py, self.states[k][1][i].py),
                    #                                color=human_colors[i], ls='solid')
                    #                     for i in range(self.human_num)]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)
            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)
            ax.set_xlim(-self.panel_width/2 - 1, self.panel_width/2 + 1)
            ax.set_ylim(-self.panel_height/2-0.5, self.panel_height/2+0.5)
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = False
            robot_color = 'black'
            # add human start positions and goals
            human_colors = [cmap(20) for i in range(len(self.humans))]
            if False:
                for i in range(len(self.humans)):
                    human = self.humans[i]
                    human_goal = mlines.Line2D([human.get_goal_position()[0]], [human.get_goal_position()[1]],
                                               color=human_colors[i],
                                               marker='*', linestyle='None', markersize=8)
                    ax.add_artist(human_goal)
                    human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                                                color=human_colors[i],
                                                marker='o', linestyle='None', markersize=8)
                    ax.add_artist(human_start)
            # add robot start position
            robot_start = mlines.Line2D([self.robot.get_start_position()[0]], [self.robot.get_start_position()[1]],
                                        color=robot_color,
                                        marker='o', linestyle='None', markersize=15)
            robot_start_position = [self.robot.get_start_position()[0], self.robot.get_start_position()[1]]
            ax.add_artist(robot_start)
            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                 color='red', marker='*', linestyle='None',
                                 markersize=15, label='Goal')
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color)
            for i in range(len(self.obstacles)):
                obstacle = self.obstacles[i]
                obstacle_mark = plt.Circle(obstacle.get_position(), obstacle.radius, fill=True, color='grey')
                ax.add_artist(obstacle_mark)

            for i in range(len(self.walls)):
                wall = self.walls[i]
                wall_line = mlines.Line2D([wall.sx, wall.ex], [wall.sy, wall.ey], color='black', marker='.',linestyle='solid', markersize=5)
                ax.add_artist(wall_line)
            # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
            ax.add_artist(robot)
            ax.add_artist(goal)
            direction_length = 1.0
            if len(self.humans) == 0:
                if display_numbers:
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        attentions =[plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                              '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)]
                # add time annotation
                time = plt.text(0.4, 1.02, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
                ax.add_artist(time)
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics =='holonomic' or i!=0:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + direction_length * radius * np.cos(theta),
                                                                            agent_state.py + direction_length * radius * np.sin(theta)))
                        else:
                            direction = (
                            (agent_state.px, agent_state.py), (agent_state.px + direction_length * radius * np.cos(agent_state.theta),
                                                               agent_state.py + direction_length * radius * np.sin(agent_state.theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0
            else:
                # add humans and their numbers
                human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
                humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False, color=human_colors[i])
                          for i in range(len(self.humans))]
                plt.legend([robot, humans[0], goal], ['Robot', 'Human', 'Goal'], fontsize=14)
                # disable showing human numbers
                if display_numbers:
                    human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i+1),
                                              color='black', fontsize=12) for i in range(len(self.humans))]
                    if hasattr(self.robot.policy, 'get_attention_weights'):
                        if self.test_changing_size is True:
                            robot_attention = [plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                        '{:.2f}'.format(self.attention_weights[0][0]), color='black',
                                                        fontsize=12)]
                            human_attentions = []
                            count = 0
                            for i in range(len(self.humans)):
                                human = humans[i]
                                dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                            human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                    human_attentions = human_attentions + [
                                        plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 '{:.2f}'.format(self.attention_weights[0][count + 1]),
                                                 color='black', fontsize=12)]
                                    count = count + 1
                                else:
                                    human_attentions = human_attentions + [
                                        plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                 'n',
                                                 color='red', fontsize=12)]
                            attentions = robot_attention + human_attentions
                        else:
                            attentions =[plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                  '{:.2f}'.format(self.attention_weights[0][0]),color='black',fontsize=12)] + \
                                        [plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset, '{:.2f}'.format(self.attention_weights[0][i+1]),
                                      color='black',fontsize=12) for i in range(len(self.humans))]
                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    if display_numbers:
                        ax.add_artist(human_numbers[i])

                # add time annotation
                time = plt.text(0.4, 1.02, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
                ax.add_artist(time)

                # visualize attention scores
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                #     attention_scores = [plt.text(-5.5, 5, 'robot {}: {:.2f}'.format(0, self.attention_weights[0][0]),
                #                  fontsize=16)] + [plt.text(-5.5, 5 - 0.5 * (i+1), 'Human {}: {:.2f}'.format(i+1, self.attention_weights[0][i+1]),
                #                  fontsize=16) for i in range(len(self.humans))]

                # compute orientation in each step and use arrow to show the direction
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics =='holonomic' or i!=0:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + direction_length * radius * np.cos(theta),
                                                                            agent_state.py + direction_length * radius * np.sin(theta)))
                        else:
                            direction = ((agent_state.px, agent_state.py), (agent_state.px + direction_length * radius * np.cos(agent_state.theta),
                                                               agent_state.py +direction_length * radius * np.sin(agent_state.theta)))
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color, arrowstyle=arrow_style)]
                    else:
                        human_arrow_color = 'red'
                        arrows.extend(
                            [patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)])

                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0

            # if len(self.trajs) != 0:
            #     human_future_positions = []
            #     human_future_circles = []
            #     for traj in self.trajs:
            #         human_future_position = [[tensor_to_joint_state(traj[step+1][0]).human_states[i].position
            #                                   for step in range(self.robot.policy.planning_depth)]
            #                                  for i in range(self.human_num)]
            #         human_future_positions.append(human_future_position)
            #
            #     for i in range(self.human_num):
            #         circles = []
            #         for j in range(self.robot.policy.planning_depth):
            #             circle = plt.Circle(human_future_positions[0][i][j], self.humans[0].radius/(1.7+j), fill=False, color=cmap(i))
            #             ax.add_artist(circle)
            #             circles.append(circle)
            #         human_future_circles.append(circles)

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                # nonlocal scores
                global_step = frame_num
                robot.center = robot_positions[frame_num]
                if self.human_num >0:
                    for i, human in enumerate(humans):
                        human.center = human_positions[frame_num][i]
                        if display_numbers:
                            human_numbers[i].set_position((human.center[0], human.center[1]))
                # if hasattr(self.robot.policy, 'get_attention_weights'):
                    # self_attention_scores = [plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
                    #                                   '{:.2f}'.format(self.attention_weights[0][0]), color='black')]
                if hasattr(self.robot.policy, 'get_attention_weights'):
                    human_attentions = []
                    count = 0
                    for i in range(self.human_num + 1):
                        if i ==0:
                            attentions[i].set_position((robot.center[0]- 0.05, robot.center[1] - x_offset))
                            attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
                        else:
                            if self.test_changing_size is True:
                                human = humans[i-1]
                                dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                        human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                    attentions[i].set_position(
                                        (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                    attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][count]))
                                    attentions[i].set_color('black')
                                else:
                                    attentions[i].set_position(
                                        (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                    attentions[i].set_text('n')
                                    attentions[i].set_color('red')
                            else:
                                attentions[i].set_position(
                                    (humans[i - 1].center[0] - 0.05, humans[i - 1].center[1] - x_offset))
                                attentions[i].set_text('{:.2f}'.format(self.attention_weights[frame_num][i]))
    #                    self_attention_dis = plt.text(robot.center[0] - x_offset, robot.center[1] + y_offset,
    #                                               '{:.2f}'.format(self.attention_weights[0][0]), color='black')
    #                    ax.add_artist(self_attention_dis)

                for arrow in arrows:
                    arrow.remove()

                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=robot_arrow_color,
                                                          arrowstyle=arrow_style)]
                    else:
                        arrows.extend([patches.FancyArrowPatch(*orientation[frame_num], color=human_arrow_color,
                                                               arrowstyle=arrow_style)])
                for arrow in arrows:
                    ax.add_artist(arrow)


                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                # if len(self.trajs) != 0:
                #     for i, circles in enumerate(human_future_circles):
                #         for j, circle in enumerate(circles):
                #             circle.center = human_future_positions[global_step][i][j]

            def plot_value_heatmap():
                if self.robot.kinematics != 'holonomic':
                    print('Kinematics is not holonomic')
                    return
                # for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                #     print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                #                                              agent.vx, agent.vy, agent.theta))

                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (self.robot.policy.rotation_samples, self.robot.policy.speed_samples))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def print_matrix_A():
                # with np.printoptions(precision=3, suppress=True):
                #     print(self.As[global_step])
                h, w = self.As[global_step].shape
                print('   ' + ' '.join(['{:>5}'.format(i - 1) for i in range(w)]))
                for i in range(h):
                    print('{:<3}'.format(i-1) + ' '.join(['{:.3f}'.format(self.As[global_step][i][j]) for j in range(w)]))
                # with np.printoptions(precision=3, suppress=True):
                #     print('A is: ')
                #     print(self.As[global_step])

            def print_feat():
                with np.printoptions(precision=3, suppress=True):
                    print('feat is: ')
                    print(self.feats[global_step])

            def print_X():
                with np.printoptions(precision=3, suppress=True):
                    print('X is: ')
                    print(self.Xs[global_step])

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                    if event.key == 'a':
                        if hasattr(self.robot.policy, 'get_matrix_A'):
                            print_matrix_A()
                        if hasattr(self.robot.policy, 'get_feat'):
                            print_feat()
                        if hasattr(self.robot.policy, 'get_X'):
                            print_X()
                        # if hasattr(self.robot.policy, 'action_values'):
                        #    plot_value_heatmap()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='Me'), bitrate=1800)
                # writer = ffmpeg_writer(fps=10, metadata=(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save output file as gif if imagemagic is installed
                # anim.save(output_file, writer='imagemagic', fps=12)
            else:
                plt.show()
        elif mode == 'debug':
            def update():
                self.ax.clear()
                self.ax.tick_params(labelsize=12)
                self.ax.set_xlim(-self.panel_width / 2 - 1 , self.panel_width / 2 + 1)
                self.ax.set_ylim(-self.panel_height / 2 - 0.5, self.panel_height / 2 + 0.5)
                self.ax.set_xlabel('x(m)', fontsize=14)
                self.ax.set_ylabel('y(m)', fontsize=14)
                robot_color = 'black'
                # add human positions and goals
                human_colors = [cmap(20) for i in range(len(self.humans))]
                # add robot and its goal
                goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                     color='red', marker='*', linestyle='None',
                                     markersize=15, label='Goal')
                robot = plt.Circle(self.robot.get_position(), self.robot.radius, fill=False, color=robot_color)
                self.ax.add_artist(robot)
                self.ax.add_artist(goal)

                for i in range(len(self.obstacles)):
                    obstacle = self.obstacles[i]
                    obstacle_mark = plt.Circle(obstacle.get_position(), obstacle.radius, fill=True, color='grey')
                    self.ax.add_artist(obstacle_mark)

                for i in range(len(self.walls)):
                    wall = self.walls[i]
                    wall_line = mlines.Line2D([wall.sx, wall.ex], [wall.sy, wall.ey], color='black', marker='.',
                                              linestyle='solid', markersize=5)
                    self.ax.add_artist(wall_line)
                # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
                direction_length = 1.0

                # if len(self.humans) == 0:

                    # print('no human')
                if len(self.humans) > 0:
                    # add humans and their numbers
                    human_positions = [human.get_position() for human in self.humans]
                    humans = [plt.Circle(human_positions[i], self.humans[i].radius, fill=False, color=human_colors[i])
                              for i in range(len(self.humans))]
                    # disable showing human numbers
                    if display_numbers:
                        human_numbers = [
                            plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i + 1),
                                     color='black', fontsize=12) for i in range(len(self.humans))]
                        if hasattr(self.robot.policy, 'get_attention_weights'):
                            if self.test_changing_size is True:
                                robot_attention = [plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                            '{:.2f}'.format(self.attention_weights[0][0]),
                                                            color='black',
                                                            fontsize=12)]
                                human_attentions = []
                                count = 0
                                for i in range(len(self.humans)):
                                    human = humans[i]
                                    dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                            human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                    if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                        human_attentions = human_attentions + [
                                            plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                     '{:.2f}'.format(self.attention_weights[0][count + 1]),
                                                     color='black', fontsize=12)]
                                        count = count + 1
                                    else:
                                        human_attentions = human_attentions + [
                                            plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                     'n',
                                                     color='red', fontsize=12)]
                                attentions = robot_attention + human_attentions
                            else:
                                attentions = [plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                       '{:.2f}'.format(self.attention_weights[0][0]), color='black',
                                                       fontsize=12)] + \
                                             [plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                       '{:.2f}'.format(self.attention_weights[0][i + 1]),
                                                       color='black', fontsize=12) for i in range(len(self.humans))]
                    for i, human in enumerate(humans):
                        self.ax.add_artist(human)
                        if display_numbers:
                            self.ax.add_artist(human_numbers[i])

                    # add time annotation
                    time = plt.text(0.4, 6.02, 'Time: {}'.format(self.global_time), fontsize=16)
                    self.ax.add_artist(time)

                    # compute orientation in each step and use arrow to show the direction
                    radius = self.robot.radius
                    orientation = []
                    for i in range(self.human_num + 1):
                        agent_state = self.robot if i == 0 else self.humans[i - 1]
                        if agent_state.kinematics == 'holonomic' or i != 0:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py),
                                         (agent_state.px + direction_length * radius * np.cos(theta),
                                          agent_state.py + direction_length * radius * np.sin(theta)))
                            orientation.append(direction)
                        else:
                            direction = ((agent_state.px, agent_state.py),
                                         (agent_state.px + direction_length * radius * np.cos(agent_state.theta),
                                          agent_state.py + direction_length * radius * np.sin(agent_state.theta)))
                            orientation.append(direction)
                        if i == 0:
                            robot_arrow_color = 'red'
                            arrows = [
                                patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color,
                                                        arrowstyle=arrow_style)]
                        else:
                            human_arrow_color = 'black'
                            arrows.extend(
                                [patches.FancyArrowPatch(*orientation[i], color=human_arrow_color,
                                                         arrowstyle=arrow_style)])
                    for arrow in arrows:
                        self.ax.add_artist(arrow)
            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(7, 7))
                self.ax.tick_params(labelsize=12)
                self.ax.set_xlim(-self.panel_width / 2, self.panel_width / 2)
                self.ax.set_ylim(-self.panel_height / 2 - 0.5, self.panel_height / 2 + 0.5)
                self.ax.set_xlabel('x(m)', fontsize=14)
                self.ax.set_ylabel('y(m)', fontsize=14)
                plt.ion()
                show_human_start_goal = False
                robot_color = 'black'
                # add human positions and goals
                human_colors = [cmap(20) for i in range(len(self.humans))]
                # add robot and its goal

                robot_positions = self.robot.get_position()
                goal = mlines.Line2D([self.robot.get_goal_position()[0]], [self.robot.get_goal_position()[1]],
                                     color='red', marker='*', linestyle='None',
                                     markersize=15, label='Goal')
                robot = plt.Circle(self.robot.get_position(), self.robot.radius, fill=False, color=robot_color)
                self.ax.add_artist(robot)
                self.ax.add_artist(goal)

                for i in range(len(self.obstacles)):
                    obstacle = self.obstacles[i]
                    obstacle_mark = plt.Circle(obstacle.get_position(), obstacle.radius, fill=True, color='grey')
                    self.ax.add_artist(obstacle_mark)

                for i in range(len(self.walls)):
                    wall = self.walls[i]
                    wall_line = mlines.Line2D([wall.sx, wall.ex], [wall.sy, wall.ey], color='black', marker='.',
                                              linestyle='solid', markersize=5)
                    self.ax.add_artist(wall_line)
                # sensor_range = plt.Circle(robot_positions[0], self.robot_sensor_range, fill=False, ls='dashed')
                direction_length = 1.0

                if len(self.humans) == 0:
                    print('no human')
                else:
                    # add humans and their numbers
                    human_positions = [human.get_position() for human in self.humans]
                    humans = [plt.Circle(human_positions[i], self.humans[i].radius, fill=False, color=human_colors[i])
                              for i in range(len(self.humans))]
                    # disable showing human numbers
                    if display_numbers:
                        human_numbers = [
                            plt.text(humans[i].center[0] - x_offset, humans[i].center[1] + y_offset, str(i + 1),
                                     color='black', fontsize=12) for i in range(len(self.humans))]
                        if hasattr(self.robot.policy, 'get_attention_weights'):
                            if self.test_changing_size is True:
                                robot_attention = [plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                            '{:.2f}'.format(self.attention_weights[0][0]),
                                                            color='black',
                                                            fontsize=12)]
                                human_attentions = []
                                count = 0
                                for i in range(len(self.humans)):
                                    human = humans[i]
                                    dis2 = (human.center[0] - robot.center[0]) * (human.center[0] - robot.center[0]) + (
                                            human.center[1] - robot.center[1]) * (human.center[1] - robot.center[1])
                                    if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                                        human_attentions = human_attentions + [
                                            plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                     '{:.2f}'.format(self.attention_weights[0][count + 1]),
                                                     color='black', fontsize=12)]
                                        count = count + 1
                                    else:
                                        human_attentions = human_attentions + [
                                            plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                     'n',
                                                     color='red', fontsize=12)]
                                attentions = robot_attention + human_attentions
                            else:
                                attentions = [plt.text(robot.center[0] + x_offset, robot.center[1] + y_offset,
                                                       '{:.2f}'.format(self.attention_weights[0][0]), color='black',
                                                       fontsize=12)] + \
                                             [plt.text(humans[i].center[0] + x_offset, humans[i].center[1] + y_offset,
                                                       '{:.2f}'.format(self.attention_weights[0][i + 1]),
                                                       color='black', fontsize=12) for i in range(len(self.humans))]
                    for i, human in enumerate(humans):
                        self.ax.add_artist(human)
                        if display_numbers:
                            self.ax.add_artist(human_numbers[i])

                    # add time annotation
                    time = plt.text(0.4, 1.02, 'Time: {}'.format(self.global_time), fontsize=16)
                    self.ax.add_artist(time)

                    # compute orientation in each step and use arrow to show the direction
                    radius = self.robot.radius
                    orientation = []
                    for i in range(self.human_num + 1):
                        agent_state = self.robot if i == 0 else self.humans[i - 1]
                        if agent_state.kinematics == 'holonomic' or i != 0:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = ((agent_state.px, agent_state.py),
                                         (agent_state.px + direction_length * radius * np.cos(theta),
                                          agent_state.py + direction_length * radius * np.sin(theta)))
                            orientation.append(direction)
                        else:
                            direction = ((agent_state.px, agent_state.py),
                                         (agent_state.px + direction_length * radius * np.cos(agent_state.theta),
                                          agent_state.py + direction_length * radius * np.sin(agent_state.theta)))
                            orientation.append(direction)
                        if i == 0:
                            robot_arrow_color = 'red'
                            arrows = [
                                patches.FancyArrowPatch(*orientation[0], color=robot_arrow_color,
                                                        arrowstyle=arrow_style)]
                        else:
                            human_arrow_color = 'black'
                            arrows.extend(
                                [patches.FancyArrowPatch(*orientation[i], color=human_arrow_color,
                                                         arrowstyle=arrow_style)])
                    for arrow in arrows:
                        self.ax.add_artist(arrow)
                plt.pause(0.001)
            else:
                update()
                plt.pause(0.001)
        else:
            raise NotImplementedError
