import torch
import numpy as np
import dgl
from dgl import DGLGraph


class CrowdNavGraph():
    def __init__(self, data):
        #        ntypes = ['robot', 'human', 'obstacle', 'wall'], etypes = ['h_r', 'o_r', 'w_r', 'h_h', 'o_h', 'w_h', 'r_h']
        super(CrowdNavGraph, self).__init__()
        self.graph = None
        self.data = None
        self.robot_visible = False
        self.rels = ['h2r', 'o2r', 'w2r', 'o2h', 'w2h', 'h2h']
        rotated_data = self.rotate_state(data)
        self.initializeWithAlternativeRelational(rotated_data)

    def rotate_state(self, state):

        """
        Transform the coordinate to agent-centric.
        Input tuple include robot state tensor and human state tensor.
        robot state tensor is of size (number, state_length)(for example 1*9)
        human state tensor is of size (number, state_length)(for example 5*5)
        obstacle state tensor is of size (number, state_length)(for example 3*3)
        wall state tensor is of size (number, state_length)(for example 4*4)
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
        # 'sx', 'sy', 'ex', 'ey', radius
        #  0     1     2     3
        assert len(state[0].shape) == 2

        robot_state = state[0]
        human_state = state[1]
        obstacle_state = state[2]
        wall_state = state[3]

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
        robot_velocities = robot_state[:, 2:4]
        robot_linear_velocity = torch.sum(robot_velocities, dim=1, keepdim=True) * 0.5
        robot_angular_velocity = (robot_velocities[:, 1] - robot_velocities[:, 0]) / (2 * 0.3)
        robot_angular_velocity = robot_angular_velocity.unsqueeze(0)
        radius_r = robot_state[:, 4].unsqueeze(1)
        v_pref = robot_state[:, 7].unsqueeze(1)
        target_heading = torch.zeros_like(radius_r)
        pos_r = torch.zeros_like(robot_velocities)
        cur_heading = (robot_state[:, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
        new_robot_state = torch.cat((pos_r, robot_velocities, radius_r,
                                     target_heading, dg, v_pref, cur_heading),
                                    dim=1)

        human_positions = human_state[:, 0:2] - robot_state[:, 0:2]
        human_positions = torch.mm(human_positions, transform_matrix)
        human_velocities = human_state[:, 2:4]
        human_velocities = torch.mm(human_velocities, transform_matrix)
        human_radius = human_state[:, 4].unsqueeze(1)
        new_human_state = torch.cat((human_positions, human_velocities, human_radius), dim=1)

        wall_start_positions = wall_state[:, 0:2] - robot_state[:, 0:2]
        wall_start_positions = torch.mm(wall_start_positions, transform_matrix)
        wall_end_positions = wall_state[:, 2:4] - robot_state[:, 0:2]
        wall_end_positions = torch.mm(wall_end_positions, transform_matrix)
        wall_radius = torch.zeros((wall_state.shape[0], 1))
        new_wall_states = torch.cat((wall_start_positions, wall_end_positions, wall_radius), dim=1)

        obstacle_positions = obstacle_state[:, 0:2] - robot_state[:, 0:2]
        obstacle_positions = torch.mm(obstacle_positions, transform_matrix)
        obstacle_radius = obstacle_state[:, 2].unsqueeze(1)
        new_obstacle_states = torch.cat((obstacle_positions, obstacle_radius), dim=1)

        new_state = (new_robot_state, new_human_state, new_obstacle_states, new_wall_states)

        return new_state

    def initializeWithAlternativeRelational(self, data):

        # We create a map to store the types of the nodes. We'll use it to compute edges' types
        self.typeMap = dict()
        position_by_id = {}

        # Node Descriptor Table
        self.node_descriptor_header = ['r', 'h', 'o', 'w']

        # # Relations are integers
        # RelTensor = torch.LongTensor
        # # Normalization factors are floats
        # NormTensor = torch.Tensor
        # # Generate relations and number of relations integer (which is only accessed outside the class)
        # max_used_id = 0 # 0 for the robot
        # # Compute closest human distance
        # closest_human_distance = -1
        # Feature dimensions
        node_types_one_hot = ['robot', 'human', 'obstacle', 'wall']
        robot_metric_features = ['rob_pos_x', 'robot_pos_y', 'rob_vel_x', 'rob_vel_y', 'rob_radius', 'rob_goal_x',
                                 'rob_goal_y', 'rob_vel_pre', 'rob_ori']
        human_metric_features = ['human_pos_x', 'human_pos_y', 'human_vel_x', 'human_vel_y', 'human_radius']
        obstacle_metric_features = ['obs_pos_x', 'obs_pos_y', 'obs_radius']
        wall_metric_features = ['start_pos_x', 'start_pos_y', 'enb_pos_x', 'end_pos_y', 'wall_radius']
        all_features = node_types_one_hot + robot_metric_features + human_metric_features + obstacle_metric_features + wall_metric_features
        # Copy input data
        self.data = data
        robot_state, human_state, obstacle_state, wall_state = self.data
        feature_dimensions = len(all_features)
        robot_num = robot_state.shape[0]
        human_num = human_state.shape[0]
        obstacle_num = obstacle_state.shape[0]
        wall_num = wall_state.shape[0]
        total_node_num = robot_num + human_num + obstacle_num + wall_num

        # add human_to_robot edges
        src_human_id = torch.tensor(range(human_num)) + robot_num
        h2r_robot_id = torch.zeros_like(src_human_id)
        h2r_edge_types = torch.ones_like(h2r_robot_id) * torch.LongTensor([self.rels.index('h2r')])
        h2r_edge_norm = torch.ones_like(h2r_robot_id) * (1.0)

        # add obstacle_to_robot edges
        src_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
        o2r_robot_id = torch.zeros_like(src_obstacle_id)
        o2r_edge_types = torch.ones_like(o2r_robot_id) * torch.LongTensor([self.rels.index('o2r')])
        o2r_edge_norm = torch.ones_like(o2r_robot_id) * (1.0)

        # add wall_to_robot edges
        src_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
        w2r_robot_id = torch.zeros_like(src_wall_id)
        w2r_edge_types = torch.ones_like(w2r_robot_id) * torch.LongTensor([self.rels.index('w2r')])
        w2r_edge_norm = torch.ones_like(w2r_robot_id) * (1.0)

        o2h_obstacle_id = None
        o2h_human_id = None
        w2h_wall_id = None
        w2h_human_id = None
        h2h_src_id = None
        h2h_dst_id = None

        for j in range(human_num):
            if j == 0:
                i = j + robot_num
                # add obstacle_to_human edges
                o2h_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                o2h_human_id = torch.ones_like(src_obstacle_id) * i
                # self.add_edges(src_obstacle_id, dst_human_id, ('obstacle', 'human', 'o_h'))

                # add wall_to_human edges
                w2h_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                w2h_human_id = torch.ones_like(src_wall_id) * i
                # self.add_edges(src_wall_id, dst_human_id, ('wall', 'human', 'w_h'))

                # add human_to_human edges
                if human_num > 1:
                    # a = (list(range(i)) + list(range(i + 1, human_num)))
                    h2h_src_id = torch.tensor(list(range(i)) + list(range(i + 1, human_num)))
                    h2h_dst_id = torch.ones_like(h2h_src_id) * i
                # self.add_edges(src_human_id, dst_human_id, ('human', 'human', 'h_h'))
            else:
                i = j + robot_num
                # add obstacle_to_human edges

                b = torch.tensor(range(obstacle_num)) + robot_num + human_num
                o2h_obstacle_id = torch.cat(o2h_obstacle_id, torch.tensor(range(obstacle_num)) + robot_num + human_num)
                o2h_human_id = torch.cat(o2h_human_id, torch.ones_like(torch.tensor(range(obstacle_num))) * i)
                w2h_wall_id = torch.cat(w2h_wall_id,
                                        torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num)
                w2h_human_id = torch.cat(w2h_human_id, torch.ones_like(torch.tensor(range(wall_num))) * i)
                if human_num > 1:
                    h2h_src_id = torch.cat(h2h_src_id, torch.tensor(range(i) + range(i + 1, human_num)))
                    h2h_dst_id = torch.cat(h2h_dst_id, torch.ones_like(
                        torch.tensor(list(range(i)) + list(range(i + 1, human_num)))) * i)

        if human_num > 1:
            self.graph = dgl.heterograph({('human', 'h2r', 'robot'): (src_human_id, h2r_robot_id),
                                          ('obstacle', 'o2r', 'robot'): (src_obstacle_id, o2r_robot_id),
                                          ('wall', 'w2r', 'robot'): (src_wall_id, w2r_robot_id),
                                          ('obstacle', 'o2h', 'human'): (o2h_obstacle_id, o2h_human_id),
                                          ('wall', 'w2h', 'human'): (w2h_wall_id, w2h_human_id),
                                          ('human', 'h2h', 'human'): (h2h_src_id, h2h_dst_id)
                                          })
        else:
            # add edges
            o2h_edge_types = torch.ones_like(o2h_human_id) * torch.LongTensor([self.rels.index('o2h')])
            o2h_edge_norm = torch.ones_like(o2h_human_id) * (1.0)

            w2h_edge_types = torch.ones_like(w2h_human_id) * torch.LongTensor([self.rels.index('w2h')])
            w2h_edge_norm = torch.ones_like(w2h_human_id) * (1.0)

            edge_types = torch.cat([h2r_edge_types, o2r_edge_types, w2r_edge_types, o2h_edge_types, w2h_edge_types],
                                   dim=0)
            edge_norm = torch.cat([h2r_edge_norm, o2r_edge_norm, w2r_edge_norm, o2h_edge_norm, w2h_edge_norm], dim=0)
            edge_norm = edge_norm.unsqueeze(dim=1)

            src_id = torch.cat([src_human_id, src_obstacle_id, src_wall_id, o2h_obstacle_id, w2h_wall_id], dim=0)
            dst_id = torch.cat([h2r_robot_id, o2r_robot_id, w2r_robot_id, o2h_human_id, w2h_human_id], dim=0)
            self.graph = dgl.graph((src_id, dst_id), num_nodes=total_node_num, idtype=torch.int32, )

        # fill data into the heterographgraph
        # data of the robot
        robot_tensor = torch.zeros((robot_num, feature_dimensions))
        robot_tensor[0, all_features.index('robot')] = 1
        robot_tensor[0, all_features.index('rob_pos_x'):all_features.index("rob_ori") + 1] = robot_state[0]
        # self.graph.nodes['robot'].data['h'] = robot_tensor

        human_tensor = torch.zeros((human_num, feature_dimensions))
        for i in range(human_num):
            human_tensor[i, all_features.index('human')] = 1
            human_tensor[i, all_features.index('human_pos_x'):all_features.index("human_radius") + 1] = human_state[i]
        # self.graph.nodes['human'].data['h'] = human_tensor

        obstacle_tensor = torch.zeros((obstacle_num, feature_dimensions))
        for i in range(obstacle_num):
            obstacle_tensor[i, all_features.index('obstacle')] = 1
            obstacle_tensor[i, all_features.index('obs_pos_x'):all_features.index("obs_radius") + 1] = obstacle_state[i]
        # self.graph.nodes['obstacle'].data['h'] = obstacle_tensor

        wall_tensor = torch.zeros((wall_num, feature_dimensions))
        for i in range(wall_num):
            wall_tensor[i, all_features.index('wall')] = 1
            wall_tensor[i, all_features.index('start_pos_x'):all_features.index("wall_radius") + 1] = wall_state[i]
        # self.graph.nodes['wall'].data['h'] = wall_tensor
        features = torch.cat([robot_tensor, human_tensor, obstacle_tensor, wall_tensor], dim=0)

        self.graph.ndata['h'] = features
        self.graph.edata.update({'rel_type': edge_types, 'norm': edge_norm})
