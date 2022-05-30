import torch
import numpy as np
import dgl
from dgl import DGLGraph
from crowd_nav.utils.rvo_inter import rvo_inter
class CrowdNavGraph():
    def __init__(self, data):
        #        ntypes = ['robot', 'human', 'obstacle', 'wall'], etypes = ['h_r', 'o_r', 'w_r', 'h_h', 'o_h', 'w_h', 'r_h']
        super(CrowdNavGraph, self).__init__()
        self.graph = None
        self.data = None
        self.robot_visible = False
        self.rels = ['h2r', 'o2r', 'w2r', 'o2h', 'w2h', 'h2h']
        self.use_rvo = True
        if self.use_rvo is True:
            self.rvo_inter = rvo_inter(neighbor_region=6, neighbor_num=20, vxmax=1, vymax=1, acceler=1.0,
                                       env_train=True,
                                       exp_radius=0.0, ctime_threshold=5, ctime_line_threshold=1)
            rotated_data = self.config_rvo_state(data)
            self.build_up_graph_on_rvostate(rotated_data)
        else:
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
        assert state[2] is not None
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
        new_robot_state = torch.cat((robot_velocities, dg, v_pref, cur_heading),
                                    dim=1)
        new_state = new_robot_state
        if state[1].shape[0] > 0:
            human_positions = human_state[:, 0:2] - robot_state[:, 0:2]
            human_positions = torch.mm(human_positions, transform_matrix)
            human_velocities = human_state[:, 2:4]
            human_velocities = torch.mm(human_velocities, transform_matrix)
            human_radius = human_state[:, 4].unsqueeze(1) + 0.3
            new_human_state = torch.cat((human_positions, human_velocities, human_radius), dim=1)
        else:
            new_human_state = None
        if state[2] is not None:
            obstacle_positions = obstacle_state[:, 0:2] - robot_state[:, 0:2]
            obstacle_positions = torch.mm(obstacle_positions, transform_matrix)
            obstacle_radius = obstacle_state[:, 2].unsqueeze(1) + 0.3
            new_obstacle_states = torch.cat((obstacle_positions, obstacle_radius), dim=1)
        else:
            new_obstacle_states = None
        if state[3] is not None:
            wall_start_positions = wall_state[:, 0:2] - robot_state[:, 0:2]
            wall_start_positions = torch.mm(wall_start_positions, transform_matrix)
            wall_end_positions = wall_state[:, 2:4] - robot_state[:, 0:2]
            wall_end_positions = torch.mm(wall_end_positions, transform_matrix)
            wall_radius = torch.zeros((wall_state.shape[0], 1)) + 0.3
            new_wall_states = torch.cat((wall_start_positions, wall_end_positions, wall_radius), dim=1)
        else:
            new_wall_states = None

        new_state = (new_robot_state, new_human_state, new_obstacle_states, new_wall_states)

        return new_state


    def rvo_state(self, state):
        return state
    def initializeWithAlternativeRelational(self, data):

        src_id = None
        dst_id = None
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
        # robot_metric_features = ['rob_pos_x', 'robot_pos_y', 'rob_vel_x', 'rob_vel_y', 'rob_radius', 'rob_goal_x',
        #                          'rob_goal_y', 'rob_vel_pre', 'rob_ori']
        robot_metric_features = ['rob_vel_l', 'rob_vel_r', 'dis2goal', 'rob_vel_pre', 'rob_ori']
        human_metric_features = ['human_pos_x', 'human_pos_y', 'human_vel_x', 'human_vel_y', 'human_radius']
        obstacle_metric_features = ['obs_pos_x', 'obs_pos_y', 'obs_radius']
        wall_metric_features = ['start_pos_x', 'start_pos_y', 'enb_pos_x', 'end_pos_y', 'wall_radius']
        all_features = node_types_one_hot + robot_metric_features + human_metric_features + obstacle_metric_features + wall_metric_features
        # Copy input data
        self.data = data
        robot_state, human_state, obstacle_state, wall_state = self.data
        feature_dimensions = len(all_features)
        robot_num = robot_state.shape[0]
        if human_state is not None:
            human_num = human_state.shape[0]
        else:
            human_num = 0
        if obstacle_state is not None:
            obstacle_num = obstacle_state.shape[0]
        else:
            obstacle_num = 0
        if wall_state is not None:
            wall_num = wall_state.shape[0]
        else:
            wall_num = 0
        total_node_num = robot_num + human_num + obstacle_num + wall_num
        # fill data into the heterographgraph
        # data of the robot
        robot_tensor = torch.zeros((robot_num, feature_dimensions))
        robot_tensor[0, all_features.index('robot')] = 1
        robot_tensor[0, all_features.index('rob_vel_l'):all_features.index("rob_ori") + 1] = robot_state[0]
        # self.graph.nodes['robot'].data['h'] = robot_tensor
        features = robot_tensor
        if human_num > 0:
            human_tensor = torch.zeros((human_num, feature_dimensions))
            for i in range(human_num):
                human_tensor[i, all_features.index('human')] = 1
                human_tensor[i, all_features.index('human_pos_x'):all_features.index("human_radius") + 1] = human_state[i]
            # self.graph.nodes['human'].data['h'] = human_tensor
            features = torch.cat([features, human_tensor], dim=0)

        if obstacle_num > 0:
            obstacle_tensor = torch.zeros((obstacle_num, feature_dimensions))
            for i in range(obstacle_num):
                obstacle_tensor[i, all_features.index('obstacle')] = 1
                obstacle_tensor[i, all_features.index('obs_pos_x'):all_features.index("obs_radius") + 1] = obstacle_state[i]
            # self.graph.nodes['obstacle'].data['h'] = obstacle_tensor
            features = torch.cat([features, obstacle_tensor], dim=0)
        if wall_num > 0:
            for i in range(wall_num):
                wall_tensor = torch.zeros((wall_num, feature_dimensions))
                wall_tensor[i, all_features.index('wall')] = 1
                wall_tensor[i, all_features.index('start_pos_x'):all_features.index("wall_radius") + 1] = wall_state[i]
            features = torch.cat([features, wall_tensor], dim=0)
        # self.graph.nodes['wall'].data['h'] = wall_tensor
        # features = torch.cat([robot_tensor, human_tensor, obstacle_tensor, wall_tensor], dim=0)

        ### build up edges for the social graph
        # add obstacle_to_robot edges
        src_id = torch.Tensor([])
        dst_id = torch.Tensor([])
        edge_types = torch.Tensor([])
        edge_norm = torch.Tensor([])
        # add human_to_robot edges
        if obstacle_num > 0:
            src_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
            o2r_robot_id = torch.zeros_like(src_obstacle_id)
            o2r_edge_types = torch.ones_like(o2r_robot_id) * torch.LongTensor([self.rels.index('o2r')])
            o2r_edge_norm = torch.ones_like(o2r_robot_id) * (1.0)
            src_id = src_obstacle_id
            dst_id = o2r_robot_id
            edge_types = o2r_edge_types
            edge_norm = o2r_edge_norm

        if human_num > 0:
            src_human_id = torch.tensor(range(human_num)) + robot_num
            h2r_robot_id = torch.zeros_like(src_human_id)
            h2r_edge_types = torch.ones_like(h2r_robot_id) * torch.LongTensor([self.rels.index('h2r')])
            h2r_edge_norm = torch.ones_like(h2r_robot_id) * (1.0)
            src_id = torch.cat([src_id, src_human_id], dim=0)
            dst_id = torch.cat([dst_id, h2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, h2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, h2r_edge_norm], dim=0)

        # add wall_to_robot edges
        if wall_num > 0:
            src_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
            w2r_robot_id = torch.zeros_like(src_wall_id)
            w2r_edge_types = torch.ones_like(w2r_robot_id) * torch.LongTensor([self.rels.index('w2r')])
            w2r_edge_norm = torch.ones_like(w2r_robot_id) * (1.0)

            src_id = torch.cat([src_id, src_wall_id], dim=0)
            dst_id = torch.cat([dst_id, w2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, w2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, w2r_edge_norm], dim=0)


        if human_num > 0:
            o2h_obstacle_id = None
            o2h_human_id = None
            w2h_wall_id = None
            w2h_human_id = None
            h2h_src_id = None
            h2h_dst_id = None
            for j in range(human_num):
                if j == 0:
                    i = j + robot_num
                    if obstacle_num > 0:
                    # add obstacle_to_human edges
                        o2h_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                        o2h_human_id = torch.ones_like(src_obstacle_id) * i
                        o2h_edge_types = torch.ones_like(o2h_human_id) * torch.LongTensor([self.rels.index('o2h')])
                        o2h_edge_norm = torch.ones_like(o2h_human_id) * (1.0)
                        src_id = torch.cat([src_id, o2h_obstacle_id], dim=0)
                        dst_id = torch.cat([dst_id, o2h_human_id], dim=0)
                        edge_types = torch.cat([edge_types, o2h_edge_types], dim=0)
                        edge_norm = torch.cat([edge_norm, o2h_edge_norm], dim=0)

                    if wall_num > 0:
                        # add wall_to_human edges
                        w2h_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                        w2h_human_id = torch.ones_like(src_wall_id) * i
                        w2h_edge_types = torch.ones_like(w2h_human_id) * torch.LongTensor([self.rels.index('w2h')])
                        w2h_edge_norm = torch.ones_like(w2h_human_id) * (1.0)
                        src_id = torch.cat([src_id, w2h_wall_id], dim=0)
                        dst_id = torch.cat([dst_id, w2h_human_id], dim=0)
                        edge_types = torch.cat([edge_types, w2h_edge_types], dim=0)
                        edge_norm = torch.cat([edge_norm, w2h_edge_norm], dim=0)
                        # self.add_edges(src_wall_id, dst_human_id, ('wall', 'human', 'w_h'))

        if human_num > 1:
            # add human_to_human edges
            temp_src_id = []
            temp_dst_id = []
            for i in range(human_num):
                for k in range(j + 1, human_num):
                # a = (list(range(i)) + list(range(i + 1, human_num)))
                    temp_src_id.append(i+robot_num)
                    temp_src_id.append(k + robot_num)
                    temp_dst_id.append(k + robot_num)
                    temp_dst_id.append(i+robot_num)
            temp_src_id = torch.IntTensor(temp_src_id)
            temp_dst_id = torch.IntTensor(temp_dst_id)
            h2h_src_id = torch.IntTensor(temp_src_id)
            h2h_dst_id = torch.IntTensor(temp_dst_id)
            h2h_edge_types = torch.ones_like(h2h_src_id) * torch.LongTensor([self.rels.index('h2h')])
            h2h_edge_norm = torch.ones_like(h2h_src_id) * (1.0)
            src_id = torch.cat([src_id, h2h_src_id], dim=0)
            dst_id = torch.cat([dst_id, h2h_dst_id], dim=0)
            edge_types = torch.cat([edge_types, h2h_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, h2h_edge_norm], dim=0)
        edge_norm = edge_norm.unsqueeze(dim=1)
        self.graph = dgl.graph((src_id, dst_id), num_nodes=total_node_num, idtype=torch.int32)
        self.graph.ndata['h'] = features
        self.graph.edata.update({'rel_type': edge_types, 'norm': edge_norm})
        # self.graph = dgl.add_self_loop(self.graph)

    def config_rvo_state(self, state):

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
        assert state[2] is not None
        robot_state = state[0]
        human_state = state[1]
        obstacle_state = state[2]
        wall_state = state[3]

        robot_state_array = robot_state.numpy()
        human_state_array = human_state.numpy()
        obstacle_state_array = obstacle_state.numpy()
        wall_state_array = wall_state.numpy()
        rvo_human_state, rvo_obstacle_state, rvo_wall_state, _, min_exp_time, _ = self.rvo_inter.config_vo_inf(robot_state_array,human_state_array, obstacle_state_array, wall_state_array)
        rvo_human_state = torch.Tensor(rvo_human_state)
        rvo_obstacle_state = torch.Tensor(rvo_obstacle_state)
        rvo_wall_state = torch.Tensor(rvo_wall_state)
        rvo_robot_state, rvo_human_state, rvo_obstacle_state, rvo_wall_state = self.world2robotframe(robot_state, rvo_human_state, rvo_obstacle_state, rvo_wall_state)
        return rvo_robot_state, rvo_human_state, rvo_obstacle_state, rvo_wall_state

    def world2robotframe(self, robot_state, human_state, obstacle_state, wall_state):
        dx = robot_state[:, 5] - robot_state[:, 0]
        dy = robot_state[:, 6] - robot_state[:, 1]
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)
        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        rot = torch.atan2(dy, dx)
        robot_velocities = robot_state[:, 2:4]
        v_pref = robot_state[:, 7].unsqueeze(1)
        robot_radius = robot_state[:, 4].unsqueeze(1)
        cur_heading = (robot_state[:, 8].unsqueeze(1) - rot + np.pi) % (2 * np.pi) - np.pi
        new_robot_state = torch.cat((robot_velocities, dg, v_pref, cur_heading, robot_radius), dim=1)

        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        transform_matrix = torch.cat((cos_rot, -sin_rot, sin_rot, cos_rot), dim=0).reshape(2, 2)
        if human_state.shape[0] != 0:
            human_state[:,8:10] = torch.mm(human_state[:,8:10], transform_matrix)
        if obstacle_state.shape[0] !=0:
            obstacle_state[:, 8:10] = torch.mm(obstacle_state[:, 8:10], transform_matrix)
        if wall_state.shape[0] != 0:
            wall_state[:,8:10] = torch.mm(wall_state[:, 8:10], transform_matrix)
            wall_state[:,10:12] = torch.mm(wall_state[:, 10:12], transform_matrix)
        return new_robot_state, human_state, obstacle_state, wall_state

    def build_up_graph_on_rvostate(self, data):

        src_id = torch.Tensor([])
        dst_id = torch.Tensor([])
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
        # robot_metric_features = ['rob_pos_x', 'robot_pos_y', 'rob_vel_x', 'rob_vel_y', 'rob_radius', 'rob_goal_x',
        #                          'rob_goal_y', 'rob_vel_pre', 'rob_ori']
        robot_metric_features = ['rob_vel_l', 'rob_vel_r', 'dis2goal', 'rob_vel_pre', 'rob_ori', 'rob_radius']
        human_metric_features = ['human_vo_px', 'human_vo_py', 'human_vo_vl_x', 'human_v0_vl_y', 'human_vo_vr_x',
                                 'human_vo_vr_y', 'human_min_dis', 'human_exp_time', 'human_pos_x', 'human_pos_y', 'human_radius']
        obstacle_metric_features = ['obs_vo_px', 'obs_vo_py', 'obs_vo_vl_x', 'obs_v0_vl_y', 'obs_vo_vr_x',
                                    'obs_vo_vr_y', 'obs_min_dis', 'obs_exp_time', 'obs_pos_x', 'obs_pos_y', 'obs_radius']
        wall_metric_features = ['wall_vo_px', 'wall_vo_py', 'wall_vo_vl_x', 'wall_v0_vl_y', 'wall_vo_vr_x',
                                'wall_vo_vr_y', 'wall_min_dis', 'wall_exp_time', 'wall_sx', 'wall_sy', 'wall_ex', 'wall_ey']
        all_features = node_types_one_hot + robot_metric_features + human_metric_features + obstacle_metric_features + wall_metric_features
        # Copy input data
        self.data = data
        robot_state, human_state, obstacle_state, wall_state = self.data
        feature_dimensions = len(all_features)
        robot_num = robot_state.shape[0]
        if human_state is not None:
            human_num = human_state.shape[0]
        else:
            human_num = 0
        if obstacle_state is not None:
            obstacle_num = obstacle_state.shape[0]
        else:
            obstacle_num = 0
        if wall_state is not None:
            wall_num = wall_state.shape[0]
        else:
            wall_num = 0
        total_node_num = robot_num + human_num + obstacle_num + wall_num
        # if total_node_num == 1:
        # fill data into the heterographgraph
        # data of the robot
        robot_tensor = torch.zeros((robot_num, feature_dimensions))
        robot_tensor[0, all_features.index('robot')] = 1
        robot_tensor[0, all_features.index('rob_vel_l'):all_features.index("rob_radius") + 1] = robot_state[0]
        # self.graph.nodes['robot'].data['h'] = robot_tensor
        features = robot_tensor
        if human_num > 0:
            human_tensor = torch.zeros((human_num, feature_dimensions))
            for i in range(human_num):
                human_tensor[i, all_features.index('human')] = 1
                human_tensor[i, all_features.index('human_vo_px'):all_features.index("human_radius") + 1] = human_state[i]
            # self.graph.nodes['human'].data['h'] = human_tensor
            features = torch.cat([features, human_tensor], dim=0)

        if obstacle_num > 0:
            obstacle_tensor = torch.zeros((obstacle_num, feature_dimensions))
            for i in range(obstacle_num):
                obstacle_tensor[i, all_features.index('obstacle')] = 1
                obstacle_tensor[i, all_features.index('obs_vo_px'):all_features.index("obs_radius") + 1] = \
                    obstacle_state[i]
            # self.graph.nodes['obstacle'].data['h'] = obstacle_tensor
            features = torch.cat([features, obstacle_tensor], dim=0)
        if wall_num > 0:
            for i in range(wall_num):
                wall_tensor = torch.zeros((wall_num, feature_dimensions))
                wall_tensor[i, all_features.index('wall')] = 1
                wall_tensor[i, all_features.index('wall_vo_px'):all_features.index("wall_ey") + 1] = wall_state[i]
            features = torch.cat([features, wall_tensor], dim=0)
        # self.graph.nodes['wall'].data['h'] = wall_tensor
        # features = torch.cat([robot_tensor, human_tensor, obstacle_tensor, wall_tensor], dim=0)

        ### build up edges for the social graph
        # add obstacle_to_robot edges
        src_id = torch.Tensor([])
        dst_id = torch.Tensor([])
        edge_types = torch.Tensor([])
        edge_norm = torch.Tensor([])
        # add human_to_robot edges

        if obstacle_num > 0:
            src_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
            o2r_robot_id = torch.zeros_like(src_obstacle_id)
            o2r_edge_types = torch.ones_like(o2r_robot_id) * torch.LongTensor([self.rels.index('o2r')])
            o2r_edge_norm = torch.ones_like(o2r_robot_id) * (1.0)
            src_id = src_obstacle_id
            dst_id = o2r_robot_id
            edge_types = o2r_edge_types
            edge_norm = o2r_edge_norm

        if human_num > 0:
            src_human_id = torch.tensor(range(human_num)) + robot_num
            h2r_robot_id = torch.zeros_like(src_human_id)
            h2r_edge_types = torch.ones_like(h2r_robot_id) * torch.LongTensor([self.rels.index('h2r')])
            h2r_edge_norm = torch.ones_like(h2r_robot_id) * (1.0)
            src_id = torch.cat([src_id, src_human_id], dim=0)
            dst_id = torch.cat([dst_id, h2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, h2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, h2r_edge_norm], dim=0)

        # add wall_to_robot edges
        if wall_num > 0:
            src_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
            w2r_robot_id = torch.zeros_like(src_wall_id)
            w2r_edge_types = torch.ones_like(w2r_robot_id) * torch.LongTensor([self.rels.index('w2r')])
            w2r_edge_norm = torch.ones_like(w2r_robot_id) * (1.0)

            src_id = torch.cat([src_id, src_wall_id], dim=0)
            dst_id = torch.cat([dst_id, w2r_robot_id], dim=0)
            edge_types = torch.cat([edge_types, w2r_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, w2r_edge_norm], dim=0)

        if human_num > 0:
            for j in range(human_num):
                if j == 0:
                    i = j + robot_num
                    if obstacle_num > 0:
                        # add obstacle_to_human edges
                        o2h_obstacle_id = torch.tensor(range(obstacle_num)) + robot_num + human_num
                        o2h_human_id = torch.ones_like(src_obstacle_id) * i
                        o2h_edge_types = torch.ones_like(o2h_human_id) * torch.LongTensor([self.rels.index('o2h')])
                        o2h_edge_norm = torch.ones_like(o2h_human_id) * (1.0)
                        src_id = torch.cat([src_id, o2h_obstacle_id], dim=0)
                        dst_id = torch.cat([dst_id, o2h_human_id], dim=0)
                        edge_types = torch.cat([edge_types, o2h_edge_types], dim=0)
                        edge_norm = torch.cat([edge_norm, o2h_edge_norm], dim=0)

                    if wall_num > 0:
                        # add wall_to_human edges
                        w2h_wall_id = torch.tensor(range(wall_num)) + robot_num + human_num + obstacle_num
                        w2h_human_id = torch.ones_like(src_wall_id) * i
                        w2h_edge_types = torch.ones_like(w2h_human_id) * torch.LongTensor([self.rels.index('w2h')])
                        w2h_edge_norm = torch.ones_like(w2h_human_id) * (1.0)
                        src_id = torch.cat([src_id, w2h_wall_id], dim=0)
                        dst_id = torch.cat([dst_id, w2h_human_id], dim=0)
                        edge_types = torch.cat([edge_types, w2h_edge_types], dim=0)
                        edge_norm = torch.cat([edge_norm, w2h_edge_norm], dim=0)
                        # self.add_edges(src_wall_id, dst_human_id, ('wall', 'human', 'w_h'))

        if human_num > 1:
            # add human_to_human edges
            temp_src_id = []
            temp_dst_id = []
            for i in range(human_num):
                for k in range(j + 1, human_num):
                    # a = (list(range(i)) + list(range(i + 1, human_num)))
                    temp_src_id.append(i + robot_num)
                    temp_src_id.append(k + robot_num)
                    temp_dst_id.append(k + robot_num)
                    temp_dst_id.append(i + robot_num)
            temp_src_id = torch.IntTensor(temp_src_id)
            temp_dst_id = torch.IntTensor(temp_dst_id)
            h2h_src_id = torch.IntTensor(temp_src_id)
            h2h_dst_id = torch.IntTensor(temp_dst_id)
            h2h_edge_types = torch.ones_like(h2h_src_id) * torch.LongTensor([self.rels.index('h2h')])
            h2h_edge_norm = torch.ones_like(h2h_src_id) * (1.0)
            src_id = torch.cat([src_id, h2h_src_id], dim=0)
            dst_id = torch.cat([dst_id, h2h_dst_id], dim=0)
            edge_types = torch.cat([edge_types, h2h_edge_types], dim=0)
            edge_norm = torch.cat([edge_norm, h2h_edge_norm], dim=0)
        edge_norm = edge_norm.unsqueeze(dim=1)
        edge_norm = edge_norm.float()
        edge_types = edge_types.float()
        self.graph = dgl.graph((src_id, dst_id), num_nodes=total_node_num, idtype=torch.int64)
        self.graph.ndata['h'] = features
        self.graph.edata.update({'rel_type': edge_types, 'norm': edge_norm})