import sys
import json
import numpy as np
import copy
from collections import namedtuple
import math
import torch
import dgl
from dgl import DGLGraph


class SocNavGraph(DGLGraph):
    def __init__(self, data, alt):
        super(SocNavGraph, self).__init__()
        self.data = None
        self.labels = None
        self.features = None
        self.num_rels = -1

        self.set_n_initializer(dgl.init.zero_initializer)
        self.set_e_initializer(dgl.init.zero_initializer)

        if alt == 'raw':
            self.initializeWithRawData(data)
        elif alt == '1':
            self.initializeWithAlternative1(data)
        elif alt == '2':
            self.initializeWithAlternative2(data)
        elif alt == '3':
            self.initializeWithAlternative3(data)
        elif alt == '4':
            self.initializeWithAlternative4(data)
        elif alt == 'relational':
            self.initializeWithAlternativeRelational(data)
        else:
            sys.exit(-1)


    def initializeWithRawData(self, data):
        '''
        This is the most simple alternative to generate a graph given the data. In fact, the walls are not even
        used.
        '''
        #Initialise typeMap
        self.typeMap = dict()
        self.typeMap[0] = 'r'

        # Node Descriptor Table
        self.node_descriptor_header = ['R', 'H', 'O', 'I',
                                       'x', 'y', 'orientation' ]

        # Feature dimensions
        node_types_one_shot = ['robot', 'human', 'object', 'interaction' ]
        metric_features = ['x', 'y', 'orientation']
        feature_dimensions = len(node_types_one_shot) + len(metric_features)

        # Copy input data
        self.data = copy.deepcopy(data)
        # Compute the number of nodes (links with information are created as nodes too)
        n_nodes = 1+len(self.data['humans'])+len(self.data['objects'])+len(self.data['links'])
        # Create the tensors
        self.features = np.zeros([n_nodes, feature_dimensions])
        self.labels = np.zeros([1, 1])
        # Generate the graph itself and fill tensor's data
        self.add_nodes(n_nodes)
        # print(minId, self.data['links'])
        max_used_id = 0
        self.labels[0][0] = float(self.data['score'])/100.
        # robot
        self.features[0, :] = np.array([1.,0.,0.,0.,  0., 0., 0.])
        for h in self.data['humans']:
            self.add_edge(h['id'], 0)
            max_used_id = max(h['id'], max_used_id)
            self.typeMap[h['id']] = 'p'
            xpos = float(h['xPos'])/100.
            ypos = float(h['yPos'])/100.
            orientation = float(h['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[h['id'], :] = np.array([0.,1.,0.,0.,   xpos, ypos, orientation])
        for o in self.data['objects']:
            self.add_edge(o['id'], 0)
            max_used_id = max(o['id'], max_used_id)
            self.typeMap[o['id']] = 'o'
            xpos = float(o['xPos'])/100.
            ypos = float(o['yPos'])/100.
            orientation = float(o['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[o['id'], :] = np.array([0.,0.,1.,0.,   xpos, ypos, orientation])

        for link in self.data['links']:
            max_used_id += 1
            link_id = max_used_id
            link.append(link_id)
            self.typeMap[link_id] = 'i'
            self.add_edge(link[1], link_id)
            self.add_edge(link_id, link[0])
            self.features[link_id, :] = np.array([0.,0.,0.,1.,   0., 0., 0.])
        self.add_edges(self.nodes(), self.nodes())

    def initializeWithAlternative1(self, data):
        self.typeMap = dict()
        self.typeMap[0] = 'r'

        # Node Descriptor Table
        self.node_descriptor_header = ['R', 'H', 'O', 'I',
                                  'dist', 'angle', 'orientation' ]


        # Feature dimensions
        node_types_one_shot = ['robot', 'human', 'object', 'interaction' ]
        metric_features = ['distance', 'angle', 'orientation']
        feature_dimensions = len(node_types_one_shot) + len(metric_features)
        # Copy input data
        self.data = copy.deepcopy(data)
        # Compute the number of nodes (links with information are created as nodes too)
        n_nodes = 1+len(self.data['humans'])+len(self.data['objects'])+len(self.data['links'])
        # Create the tensors
        self.features = np.zeros([n_nodes, feature_dimensions])
        self.labels = np.zeros([1, 1])
        # Generate the graph itself and fill tensor's data
        self.add_nodes(n_nodes)
        # print(minId, self.data['links'])
        max_used_id = 0
        self.labels[0][0] = float(self.data['score'])/100.
        # robot
        self.features[0, :] = np.array([1.,0.,0.,0.,  0., 0., 0.])
        for h in self.data['humans']:
            self.add_edge(h['id'], 0)
            max_used_id = max(h['id'], max_used_id)
            self.typeMap[h['id']] = 'p'
            xpos = float(h['xPos'])/100.
            ypos = float(h['yPos'])/100.
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(h['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[h['id'], :] = np.array([0.,1.,0.,0.,   distance, angle, orientation])
        for o in self.data['objects']:
            self.add_edge(o['id'], 0)
            max_used_id = max(o['id'], max_used_id)
            self.typeMap[o['id']] = 'o'
            xpos = float(o['xPos'])/100.
            ypos = float(o['yPos'])/100.
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(o['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[o['id'], :] = np.array([0.,0.,1.,0.,   distance, angle, orientation])

        for link in self.data['links']:
            max_used_id += 1
            link_id = max_used_id
            link.append(link_id)
            self.typeMap[link_id] = 'i'
            self.add_edge(link[1], link_id)
            self.add_edge(link_id, link[0])
            self.features[link_id, :] = np.array([0.,0.,0.,1.,   0., 0., 0.])
        self.add_edges(self.nodes(), self.nodes())

    def initializeWithAlternative2(self, data):
        self.typeMap = dict()
        self.typeMap[0] = 'r'

        # Node Descriptor Table
        self.node_descriptor_header = ['R', 'H', 'O', 'I',
                                  'dist', 'dist2', 'angle', 'orientation', 'min_human', 'number_humans' ]


        # Compute min humans
        min_human = -1
        # Feature dimensions
        node_types_one_hot = ['robot', 'human', 'object', 'interaction' ]
        metric_features = ['distance', 'distance2', 'angle', 'orientation', 'min_human', 'number_humans']
        feature_dimensions = len(node_types_one_hot) + len(metric_features)
        # Copy input data
        self.data = copy.deepcopy(data)
        # Compute the number of nodes (links with information are created as nodes too)
        n_nodes = 1+len(self.data['humans'])+len(self.data['objects'])+len(self.data['links'])
        # Create the tensors
        self.features = np.zeros([n_nodes, feature_dimensions])
        self.labels = np.zeros([1, 1])
        # Generate the graph itself and fill tensor's data
        self.add_nodes(n_nodes)
        # print(minId, self.data['links'])
        max_used_id = 0
        self.labels[0][0] = float(self.data['score'])/100.
        # humans
        for h in self.data['humans']:
            self.add_edge(h['id'], 0)
            self.typeMap[h['id']] = 'p'
            max_used_id = max(h['id'], max_used_id)
            xpos = float(h['xPos'])/100.
            ypos = float(h['yPos'])/100.
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(h['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[h['id'], :] = np.array([0.,1.,0.,0.,   distance, distance*distance, angle, orientation, 0, 0])
            if min_human<0 or min_human>distance:
                min_human = distance
        # objects
        for o in self.data['objects']:
            self.add_edge(o['id'], 0)
            self.typeMap[o['id']] = 'o'
            max_used_id = max(o['id'], max_used_id)
            xpos = float(o['xPos'])/100.
            ypos = float(o['yPos'])/100.
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(o['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[o['id'], :] = np.array([0.,0.,1.,0.,   distance, distance*distance, angle, orientation, 0, 0])
        for link in self.data['links']:
            max_used_id += 1
            link_id = max_used_id
            link.append(link_id)
            self.add_edge(link[1], link_id)
            self.add_edge(link_id, link[0])
            self.typeMap[link_id] = 'i'
            a = self.features[link[0], 4]   # 4 is the index of the distance
            b = self.features[link[1], 4]
            d = (a+b)/2.
            d2 = d*d
            a = self.features[link[0], 6]   # 6 is the index of the angle
            b = self.features[link[1], 6]
            angle = (a+b)/2.
            self.features[link_id, :] = np.array([0.,0.,0.,1.,  d, d2, angle, 0., 0, 0])

        # robot
        self.features[0, :] = np.array([1.,0.,0.,0.,    0., 0., 0., 0., min_human, len(self.data['humans'])])

        # add self edges
        self.add_edges(self.nodes(), self.nodes())

    def initializeWithAlternative3(self, data):
        self.typeMap = dict()
        self.typeMap[0] = 'r'
        position_by_id = {}

        # Node Descriptor Table
        self.node_descriptor_header = ['R', 'H', 'O', 'I', 'L', 'W',
                                       'dist', 'dist2', 'angle', 'orientation', 'min_human', 'number_humans' ]

        Wall = namedtuple('Wall', ['dist', 'angle'])
        # Initialise id counter
        max_used_id = 0
        # Compute min humans
        min_human = -1
        # Feature dimensions
        node_types_one_hot = ['robot', 'human', 'object', 'interaction', 'room', 'wall']
        metric_features = ['distance', 'distance2', 'angle', 'orientation', 'min_human', 'number_humans']
        feature_dimensions = len(node_types_one_hot) + len(metric_features)
        # Copy input data
        self.data = copy.deepcopy(data)
        # Compute data for walls
        walls = []
        for wall_index in range(len(self.data['room'])-1):
            p1 = np.array(self.data['room'][wall_index+0])
            p2 = np.array(self.data['room'][wall_index+1])
            dist = np.linalg.norm(p1-p2)
            iters = int(dist / 400)
            if iters > 0:
                v = (p2-p1)/iters
                for i in range(iters):
                    p = p1 + v*iters
                    walls.append(Wall(np.linalg.norm(p)/100., math.atan2(p[0], p[1])))
            walls.append(Wall(np.linalg.norm(p2)/100., math.atan2(p2[0], p2[1])))
        # Compute the number of nodes (links with information are created as nodes too)
        # one for the robot, room + walls      + humans                   + objects                   + nodes for links
        n_nodes =     1 + len(self.data['humans']) + len(self.data['objects']) +   1 +    len(walls) +  len(self.data['links'])
        # Create the tensors
        self.features = np.zeros([n_nodes, feature_dimensions])
        self.labels = np.zeros([1, 1])
        # Generate the graph itself and fill tensor's data
        self.add_nodes(n_nodes)
        # print(minId, self.data['links'])
        self.labels[0][0] = float(self.data['score'])/100.
        # humans
        for h in self.data['humans']:
            self.add_edge(h['id'], 0)
            max_used_id = max(h['id'], max_used_id)
            self.typeMap[h['id']] = 'p'
            xpos = float(h['xPos'])/100.
            ypos = float(h['yPos'])/100.
            position_by_id[h['id']] = [xpos, ypos]
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(h['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[h['id'], :] = np.array([0.,1.,0.,0.,0.,0.,   distance, distance*distance, angle, orientation, 0, 0])
            if min_human<0 or min_human>distance:
                min_human = distance
        # objects
        for o in self.data['objects']:
            self.add_edge(o['id'], 0)
            self.typeMap[o['id']] = 'o'
            max_used_id = max(o['id'], max_used_id)
            xpos = float(o['xPos'])/100.
            ypos = float(o['yPos'])/100.
            position_by_id[o['id']] = [xpos, ypos]
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(o['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[o['id'], :] = np.array([0.,0.,1.,0.,0.,0.,   distance, distance*distance, angle, orientation, 0, 0])
        # room
        max_used_id += 1
        room_id = max_used_id
        self.typeMap[room_id] = 'l'
        self.add_edge(room_id, 0)
        self.features[room_id, :] = np.array([0.,0.,0.,0.,1.,0.,    0., 0., 0., 0., min_human, len(self.data['humans'])])

        # walls
        for wall in walls:
            max_used_id += 1
            wall_id = max_used_id
            self.typeMap[wall_id] = 'w'
            self.add_edge(wall_id, room_id)
            self.features[wall_id, :] = np.array([0.,0.,0.,0.,0.,1.,    wall.dist, wall.dist*wall.dist, wall.angle, 0., 0., 0.])

        for link in self.data['links']:
            max_used_id += 1
            link_id = max_used_id
            link.append(link_id)
            self.typeMap[link_id] = 'i'
            self.add_edge(link[1], link_id)
            self.add_edge(link_id, link[0])
            a = self.features[link[0], 6]   # 6 is the index of the distance
            b = self.features[link[1], 6]
            d = (a+b)/2.
            d2 = d*d
            e1 = np.array(position_by_id[link[0]])
            e2 = np.array(position_by_id[link[1]])
            e = (e1 + e2) / 2.
            angle = math.atan2(e[0], e[1])
            self.features[link_id, :] = np.array([0.,0.,0.,1.,0.,0.,  d, d2, angle, 0., 0, 0])


        # robot
        self.features[0, :] = np.array([1.,0.,0.,0.,0.,0.,    0., 0., 0., 0., min_human, len(self.data['humans'])])

        # add self edges
        self.add_edges(self.nodes(), self.nodes())

    def initializeWithAlternative4(self, data):
        self.typeMap = dict()
        self.typeMap[0] = 'r'
        position_by_id = {}

        # Node Descriptor Table
        self.node_descriptor_header = ['R', 'H', 'O', 'I', 'L', 'W',
                                  'h_dist', 'h_dist2', 'h_ang', 'h_orient',
                                  'o_dist', 'o_dist2', 'o_ang', 'o_orient',
                                  'r_m_h', 'r_m_h2', 'r_hs', 'r_hs2',
                                  'i_dist', 'i_dist2', 'i_ang',
                                  'w_dist', 'w_dist2', 'w_ang' ]

        Wall = namedtuple('Wall', ['dist', 'angle'])
        # Initialise id counter
        max_used_id = 0
        # Compute min humans
        min_human = -1
        # Feature dimensions
        node_types_one_hot = ['robot', 'human', 'object', 'interaction', 'room', 'wall']
        human_metric_features = ['hum_distance', 'hum_distance2', 'hum_angle', 'hum_orientation' ]
        object_metric_features = ['obj_distance', 'obj_distance2', 'obj_angle', 'obj_orientation' ]
        room_metric_features = [ 'room_min_human',  'room_min_human2', 'room_humans', 'room_humans2' ]
        interaction_metric_features = [ 'int_distance', 'int_distance2', 'int_angle' ]
        wall_metric_features = [ 'wall_distance', 'wall_distance2', 'wall_angle' ]
        all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + interaction_metric_features + wall_metric_features
        feature_dimensions = len(all_features)
        # Copy input data
        self.data = copy.deepcopy(data)
        # Compute data for walls
        walls = []
        for wall_index in range(len(self.data['room'])-1):
            p1 = np.array(self.data['room'][wall_index+0])
            p2 = np.array(self.data['room'][wall_index+1])
            dist = np.linalg.norm(p1-p2)
            iters = int(dist / 400)
            if iters > 0:
                v = (p2-p1)/iters
                for i in range(iters):
                    p = p1 + v*iters
                    walls.append(Wall(np.linalg.norm(p)/100., math.atan2(p[0], p[1])))
            walls.append(Wall(np.linalg.norm(p2)/100., math.atan2(p2[0], p2[1])))
        # Compute the number of nodes (links with information are created as nodes too)
        # one for the robot, room + walls      + humans                   + objects                   + nodes for links
        n_nodes =     1 + len(self.data['humans']) + len(self.data['objects']) +   1 +    len(walls) +  len(self.data['links'])
        # Create the tensors
        self.features = np.zeros([n_nodes, feature_dimensions])
        self.labels = np.zeros([1, 1])
        # Generate the graph itself and fill tensor's data
        self.add_nodes(n_nodes)
        # print(minId, self.data['links'])
        self.labels[0][0] = float(self.data['score'])/100.
        # humans
        for h in self.data['humans']:
            self.add_edge(h['id'], 0)
            max_used_id = max(h['id'], max_used_id)
            self.typeMap[h['id']] = 'p'
            xpos = float(h['xPos'])/100.
            ypos = float(h['yPos'])/100.
            position_by_id[h['id']] = [xpos, ypos]
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(h['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[h['id'], all_features.index('human')] = 1.
            self.features[h['id'], all_features.index('hum_distance')] = distance
            self.features[h['id'], all_features.index('hum_distance2')] = distance*distance
            self.features[h['id'], all_features.index('hum_angle')] = angle
            self.features[h['id'], all_features.index('hum_orientation')] = orientation
            if min_human<0 or min_human>distance:
                min_human = distance
        # objects
        for o in self.data['objects']:
            self.add_edge(o['id'], 0)
            self.typeMap[o['id']] = 'o'
            max_used_id = max(o['id'], max_used_id)
            xpos = float(o['xPos'])/100.
            ypos = float(o['yPos'])/100.
            position_by_id[o['id']] = [xpos, ypos]
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(o['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[o['id'], all_features.index('object')] = 1
            self.features[o['id'], all_features.index('obj_distance')] = distance
            self.features[o['id'], all_features.index('obj_distance2')] = distance*distance
            self.features[o['id'], all_features.index('obj_angle')] = angle
            self.features[o['id'], all_features.index('obj_orientation')] = orientation
        # room
        max_used_id += 1
        room_id = max_used_id
        self.typeMap[room_id] = 'l'
        self.add_edge(room_id, 0)
        self.features[room_id, all_features.index('room')] = 1.
        self.features[room_id, all_features.index('room_min_human')] = min_human
        self.features[room_id, all_features.index('room_min_human2')] = min_human*min_human
        self.features[room_id, all_features.index('room_humans')] = len(self.data['humans'])
        self.features[room_id, all_features.index('room_humans')] = len(self.data['humans'])*len(self.data['humans'])

        # walls
        for wall in walls:
            max_used_id += 1
            wall_id = max_used_id
            self.typeMap[wall_id] = 'w'
            self.add_edge(wall_id, room_id)
            self.features[wall_id, all_features.index('wall')] = 1.
            self.features[wall_id, all_features.index('wall_distance')] = wall.dist
            self.features[wall_id, all_features.index('wall_distance2')] = wall.dist*wall.dist
            self.features[wall_id, all_features.index('wall_angle')] = wall.angle
        for link in self.data['links']:
            max_used_id += 1
            link_id = max_used_id
            link.append(link_id)
            self.typeMap[link_id] = 'i'
            self.add_edge(link[1], link_id)
            self.add_edge(link_id, link[0])
            a = self.features[link[0], 6]   # 6 is the index of the distance
            b = self.features[link[1], 6]
            d = (a+b)/2.
            d2 = d*d
            e1 = np.array(position_by_id[link[0]])
            e2 = np.array(position_by_id[link[1]])
            e = (e1 + e2) / 2.
            angle = math.atan2(e[0], e[1])
            self.features[link_id, all_features.index('interaction')] = 1
            self.features[link_id, all_features.index('int_distance')] = d
            self.features[link_id, all_features.index('int_distance2')] = d2
            self.features[link_id, all_features.index('int_angle')] = angle


        # robot
        self.features[0, all_features.index('robot')] = 1.

        # add self edges
        self.add_edges(self.nodes(), self.nodes())



    def initializeWithAlternativeRelational(self, data):
        # We create a map to store the types of the nodes. We'll use it to compute edges' types
        self.typeMap = dict()
        position_by_id = {}

        # Node Descriptor Table
        self.node_descriptor_header = ['R', 'H', 'O', 'L', 'W',
                                  'h_dist', 'h_dist2', 'h_ang', 'h_orient',
                                  'o_dist', 'o_dist2', 'o_ang', 'o_orient',
                                  'r_m_h', 'r_m_h2', 'r_hs', 'r_hs2',
                                  'w_dist', 'w_dist2', 'w_ang' ]

        # Relations are integers
        RelTensor = torch.LongTensor
        # Normalization factors are floats
        NormTensor = torch.Tensor
        # Generate relations and number of relations integer (which is only accessed outside the class)
        self.rels = set(['p_r', 'o_r', 'l_r', 'p_p', 'p_o', 'w_l'])
        for e in list(self.rels):
            self.rels.add(e[::-1])
        self.rels.add('self')
        self.rels = sorted(list(self.rels))
        self.num_rels = len(self.rels)
        # print('RELS: {}'.format(self.num_rels))
        # Initialise id counter
        max_used_id = 0 # 0 for the robot
        # Compute closest human distance
        closest_human_distance = -1
        # Feature dimensions
        node_types_one_hot = ['robot', 'human', 'object', 'room', 'wall']
        human_metric_features = ['hum_distance', 'hum_distance2', 'hum_angle', 'hum_orientation' ]
        object_metric_features = ['obj_distance', 'obj_distance2', 'obj_angle', 'obj_orientation' ]
        room_metric_features = [ 'room_min_human',  'room_min_human2', 'room_humans', 'room_humans2' ]
        wall_metric_features = [ 'wall_distance', 'wall_distance2', 'wall_angle' ]
        all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + wall_metric_features
        feature_dimensions = len(all_features)
        # Copy input data
        self.data = copy.deepcopy(data)


         # Compute data for walls
        Wall = namedtuple('Wall', ['dist', 'angle'])
        walls = []
        for wall_index in range(len(self.data['room'])-1):
            p1 = np.array(self.data['room'][wall_index+0])
            p2 = np.array(self.data['room'][wall_index+1])
            dist = np.linalg.norm(p1-p2)
            iters = int(dist / 400)
            if iters > 0:
                v = (p2-p1)/iters
                for i in range(iters):
                    p = p1 + v*iters
                    walls.append(Wall(np.linalg.norm(p)/100., math.atan2(p[0]/1000., p[1]/1000.)))
            walls.append(Wall(np.linalg.norm(p2)/100., math.atan2(p2[0], p2[1])))



        # Compute the number of nodes
        # one for the robot + room walls      + humans                   + objects
        n_nodes =     1 +    len(walls) + len(self.data['humans']) + len(self.data['objects']) +    1
        # print('Nodes {} robot(1) walls({}), humans({}), object({}) room(1)'.format(n_nodes, len(walls), len(self.data['humans']), len(self.data['objects'])))
        # Create the tensors
        self.features = np.zeros([n_nodes, feature_dimensions])
        self.labels = np.zeros([1, 1]) # A 1x1 tensor
        # Generate the graph itself and fill tensor's data
        self.add_nodes(n_nodes)
        # print(minId, self.data['links'])
        self.labels[0][0] = float(self.data['score'])/100.


        # robot
        self.typeMap[0] = 'r'
        self.features[0, all_features.index('robot')] = 1.


        # humans
        for h in self.data['humans']:
            self.add_edge(h['id'], 0, {'rel_type': RelTensor([[self.rels.index('p_r')]]), 'norm': NormTensor([[1./len(self.data['humans'])]]) })
            self.add_edge(0, h['id'], {'rel_type': RelTensor([[self.rels.index('r_p')]]), 'norm': NormTensor([[1.]]) })
            self.typeMap[h['id']] = 'p' # 'p' for 'person'
            max_used_id = max(h['id'], max_used_id)
            xpos = float(h['xPos'])/100.
            ypos = float(h['yPos'])/100.
            position_by_id[h['id']] = [xpos, ypos]
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(h['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[h['id'], all_features.index('human')] = 1.
            self.features[h['id'], all_features.index('hum_distance')] = distance
            self.features[h['id'], all_features.index('hum_distance2')] = distance*distance
            self.features[h['id'], all_features.index('hum_angle')] = angle
            self.features[h['id'], all_features.index('hum_orientation')] = orientation
            if closest_human_distance<0 or closest_human_distance>distance:
                closest_human_distance = distance

        # objects
        for o in self.data['objects']:
            self.add_edge(o['id'], 0, {'rel_type': RelTensor([[self.rels.index('o_r')]]), 'norm': NormTensor([[1./len(self.data['objects'])]]) })
            self.add_edge(0, o['id'], {'rel_type': RelTensor([[self.rels.index('r_o')]]), 'norm': NormTensor([[1.]]) })
            self.typeMap[o['id']] = 'o' # 'o' for 'object'
            max_used_id = max(o['id'], max_used_id)
            xpos = float(o['xPos'])/100.
            ypos = float(o['yPos'])/100.
            position_by_id[o['id']] = [xpos, ypos]
            distance = math.sqrt(xpos*xpos + ypos*ypos)
            angle = math.atan2(xpos, ypos)
            orientation = float(o['orientation'])/180.
            if orientation > 1.: orientation = -2.+orientation
            self.features[o['id'], all_features.index('object')] = 1
            self.features[o['id'], all_features.index('obj_distance')] = distance
            self.features[o['id'], all_features.index('obj_distance2')] = distance*distance
            self.features[o['id'], all_features.index('obj_angle')] = angle
            self.features[o['id'], all_features.index('obj_orientation')] = orientation

        # room
        max_used_id = max_used_id + 1
        room_id = max_used_id
        # print('Room will be {}'.format(room_id))
        self.typeMap[room_id] = 'l'
        self.add_edge(room_id, 0, {'rel_type': RelTensor([[self.rels.index('l_r')]]), 'norm': NormTensor([[1.]]) })
        self.add_edge(0, room_id, {'rel_type': RelTensor([[self.rels.index('r_l')]]), 'norm': NormTensor([[1.]]) })
        self.features[room_id, all_features.index('room')] = 1.
        self.features[room_id, all_features.index('room_min_human')] =  closest_human_distance
        self.features[room_id, all_features.index('room_min_human2')] = closest_human_distance*closest_human_distance
        self.features[room_id, all_features.index('room_humans')] = len(self.data['humans'])
        self.features[room_id, all_features.index('room_humans2')] = len(self.data['humans'])*len(self.data['humans'])


        # walls
        for wall in walls:
            max_used_id += 1
            wall_id = max_used_id
            self.typeMap[wall_id] = 'w' # 'w' for 'walls'
            # print('wall {} room {}'.format(wall_id, room_id))
            self.add_edge(wall_id, room_id, {'rel_type': RelTensor([[self.rels.index('w_l')]]), 'norm': NormTensor([[1./len(walls)]]) })
            self.add_edge(room_id, wall_id, {'rel_type': RelTensor([[self.rels.index('l_w')]]), 'norm': NormTensor([[1.]]) })
            self.features[wall_id, all_features.index('wall')] = 1.
            self.features[wall_id, all_features.index('wall_distance')] = wall.dist
            self.features[wall_id, all_features.index('wall_distance2')] = wall.dist*wall.dist
            self.features[wall_id, all_features.index('wall_angle')] = wall.angle

        # interaction links
        for link in self.data['links']:
            typeLdir = self.typeMap[link[0]] + '_' + self.typeMap[link[1]]
            typeLinv = self.typeMap[link[1]] + '_' + self.typeMap[link[0]]
            self.add_edge(link[0], link[1], {'rel_type': RelTensor([[self.rels.index(typeLdir)]]), 'norm': NormTensor([[1.]])})
            self.add_edge(link[1], link[0], {'rel_type': RelTensor([[self.rels.index(typeLinv)]]), 'norm': NormTensor([[1.]])})

        # TODO XXX add self edges
        #for relation_type in self.rels: # originally only applied to 'self'
        #    for i in range(n_nodes):
        #        self.add_edge(i, i, {'rel_type': RelTensor([[self.rels.index(relation_type)]]), 'norm': NormTensor([[1.]])})


class SocNavDataset(object):
    def __init__(self, path, mode, alt, init_line=-1, end_line=-1, verbose=True):
        super(SocNavDataset, self).__init__()
        self.path = path
        self.mode = mode
        self.alt = alt
        self.init_line = init_line
        self.end_line = end_line
        self.data = []
        self.num_rels = -1
        self.verbose = verbose
        self._load(alt)
        self._preprocess()



    def _load(self, alt):
        if type(self.path) is str:
            linen = -1
            for line in open(self.path).readlines():
                linen += 1
                if self.init_line >= 0 and linen < self.init_line:
                    continue
                if linen > self.end_line >= 0:
                    continue
                try:
                    self.data.append(SocNavGraph(json.loads(line), alt))
                except:
                    print(line)
                    raise
                if linen % 1000 == 0:
                    print(linen)
        else:
            self.data.append(SocNavGraph(self.path, alt))
        if self.verbose:
            print('{} scenarios loaded.'.format(len(self.data)))
        self.num_rels = self.data[0].num_rels
        self.graph = dgl.batch(self.data)
        self.features = torch.from_numpy(np.concatenate([element.features for element in self.data]))
        self.labels = torch.from_numpy(np.concatenate([element.labels for element in self.data]))

    def _preprocess(self):
        pass  # We don't really need to do anything, all pre-processing is done in the _load method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.data[item].features, self.data[item].labels
