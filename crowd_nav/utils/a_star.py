import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
class PriorityQueue(object):
    """Priority queue implemented by min heap used to hold nodes in exploring queue, which could achieve extracting node
    with least priority and inserting new node in O(log(n)) of time complexity, where n is the number of nodes in queue
    """

    def __init__(self):
        self.queue = []

    def push(self, node, cost):
        heapq.heappush(self.queue, (cost, node))

    def pop(self):
        return heapq.heappop(self.queue)[1]

    def empty(self):
        return len(self.queue) == 0


class Search(object):
    """Search methods for path planner
    """

    def __init__(self, world_state, robot_pose, goal_pose, obs_list, robot_size):
        self.world_state = world_state
        self.robot_pose = robot_pose
        self.goal_pose = goal_pose
        self.x_range = len(world_state)
        self.y_range = len(world_state[0])

        self.robot_size = robot_size  # Robot size
        self.obs_list = obs_list  # Obstacles
        self.frontier = PriorityQueue()  # Exploring queue
        self.cost = {}  # Record nodes and their costs from start pose
        self.parent = {}  # Record visitted nodes and their parents

        self.frontier.push(robot_pose, 0)
        self.cost[robot_pose] = 0
        self.parent[robot_pose] = None

    def A_star(self):
        # Optimal planner achieved by A* Algorithm
        while not self.frontier.empty():
            # Extract and visit nodes with least priority
            cur = self.frontier.pop()

            # If we reach goal pose, track back to get path
            if cur == self.goal_pose:
                return self.generate_path(cur)

            # Get possible next step movements of current node
            motions = self.get_robot_motion(cur)
            for motion in motions:
                node = motion[0]
                cost = motion[1]
                new_cost = self.cost[cur] + cost
                # No need to explore node that has been visited or its cost doesn't need to be updated
                if node not in self.parent or new_cost < self.cost[node]:
                    self.cost[node] = new_cost
                    priority = new_cost + self.cal_heuristic(node)
                    self.frontier.push(node, priority)
                    self.parent[node] = cur
        return None

    def cal_heuristic(self, node):
        # Calculate distance between node and goal_pose as heuristic
        return (node[0] - self.goal_pose[0]) ** 2 + (node[1] - self.goal_pose[1]) ** 2

    def get_robot_motion(self, node):
        # Robot motion model
        xr = self.x_range
        yr = self.y_range
        next_step = []
        robot_motion = [[(1, 0), 1], [(0, 1), 1], [(-1, 0), 1], [(0, -1), 1],
                        [(-1, -1), math.sqrt(2)], [(-1, 1), math.sqrt(2)],
                        [(1, -1), math.sqrt(2)], [(1, 1), math.sqrt(2)]]
        for motion in robot_motion:
            x = node[0] + motion[0][0]
            y = node[1] + motion[0][1]
            if x in range(xr) and y in range(yr) and not self.check_collision(x, y):
                next_step.append([(x, y), motion[1]])
        return next_step

    def check_collision(self, x, y):
        # Check if node get collision with obstacles
        for obs in self.obs_list:
            if self.obs_check(x, y, obs):
                return True
        return False

    def obs_check(self, x, y, obs):
        # Check if node get collision with obstacle, take into consideration about robot size
        if x >= obs[0][0] - self.robot_size and x < obs[0][1] + self.robot_size and y >= obs[1][
            0] - self.robot_size and y < obs[1][1] + self.robot_size:
            return True
        else:
            return False

    def generate_path(self, goal):
        # Track back to get path from robot pose to goal pose
        path = [goal]
        node = self.parent[goal]
        while node != self.robot_pose:
            path.append(node)
            node = self.parent[node]
        path.append(self.robot_pose)
        return path[::-1]

def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

class Search2(object):
    """Search methods for path planner
    """

    def __init__(self, world_state, robot_pose, goal_pose, cir_obs_list, line_obs_list, robot_size):
        self.world_state = world_state
        self.robot_pose = robot_pose
        self.goal_pose = goal_pose
        self.x_range = len(world_state)
        self.y_range = len(world_state[0])
        self.x_resolution = 10 / self.x_range
        self.y_resolution = 10 / self.y_range

        self.robot_size = robot_size  # Robot size
        self.circle_obs_list = cir_obs_list  # Obstacles
        self.line_obs_list = line_obs_list  #
        self.frontier = PriorityQueue()  # Exploring queue
        self.cost = {}  # Record nodes and their costs from start pose
        self.parent = {}  # Record visitted nodes and their parents

        self.frontier.push(robot_pose, 0)
        self.cost[robot_pose] = 0
        self.parent[robot_pose] = None

    def inverse(self, coordinate):
        x = coordinate[0] * self.x_resolution - 5.0
        y = coordinate[1] * self.y_resolution - 5.0
        return x,y

    def A_star(self):
        # Optimal planner achieved by A* Algorithm
        while not self.frontier.empty():
            # Extract and visit nodes with least priority
            cur = self.frontier.pop()
            if self.check_collision(self.goal_pose[0], self.goal_pose[1]):
                return None
            if self.check_collision(cur[0], cur[1]):
                return None
            # If we reach goal pose, track back to get path
            if cur == self.goal_pose:
                return self.generate_path(cur)

            # Get possible next step movements of current node
            motions = self.get_robot_motion(cur)
            for motion in motions:
                node = motion[0]
                cost = motion[1]
                new_cost = self.cost[cur] + cost
                # No need to explore node that has been visited or its cost doesn't need to be updated
                if node not in self.parent or new_cost < self.cost[node]:
                    self.cost[node] = new_cost
                    priority = new_cost + self.cal_heuristic(node)
                    self.frontier.push(node, priority)
                    self.parent[node] = cur
        return None

    def cal_heuristic(self, node):
        # Calculate distance between node and goal_pose as heuristic
        return np.abs(node[0] - self.goal_pose[0]) + np.abs(node[1] - self.goal_pose[1])

    def get_robot_motion(self, node):
        # Robot motion model
        xr = self.x_range
        yr = self.y_range
        next_step = []
        robot_motion = [[(1, 0), 1], [(0, 1), 1], [(-1, 0), 1], [(0, -1), 1]]
        for motion in robot_motion:
            x = node[0] + motion[0][0]
            y = node[1] + motion[0][1]
            if x in range(xr) and y in range(yr) and not self.check_collision(x, y):
                next_step.append([(x, y), motion[1]])
        return next_step

    def check_collision(self, x, y):
        # Check if node get collision with obstacles
        new_x,new_y = self.inverse((x,y))
        for obs in self.circle_obs_list:
            if self.obs_check(new_x, new_y, obs):
                return True
        for obs in self.line_obs_list:
            if self.line_obs_check(new_x,new_y,obs):
                return True
        return False

    def obs_check(self, x, y, obs):
        # Check if node get collision with obstacle, take into consideration about robot size
        if np.abs(x-obs[0]) < obs[2] and np.abs(y-obs[1]) < obs[2]:
            if (x-obs[0]) * (x-obs[0]) + (y - obs[1])*(y-obs[1]) < obs[2] * obs[2]:
                return True
        else:
            return False

    def line_obs_check(self, x, y, obs):
        # Check if node get collision with obstacle, take into consideration about robot size
        if (obs[0] == obs[2] and np.abs(x -obs[0]) < obs[4]) or (obs[1] == obs[3] and np.abs(y -obs[1]) < obs[4]):
            if point_to_segment_dist(obs[0], obs[1], obs[2], obs[3], x, y) < obs[4]:
                return True
        else:
            return False

    def generate_path(self, goal):
        # Track back to get path from robot pose to goal pose
        path = [goal]
        node = self.parent[goal]
        while node != self.robot_pose:
            path.append(node)
            node = self.parent[node]
        path.append(self.robot_pose)
        return path[::-1]

def generate_obstacle(xr, yr):
    # Obstacle generation
    obs_list = [[(0, xr), (0, 1)],
                [(0, xr), (yr - 1, yr)],
                [(0, 1), (0, yr)],
                [(xr - 1, xr), (0, yr)]]
    obs_list.append([(15, 16), (0, 30)])
    obs_list.append([(35, 36), (20, 50)])
    return obs_list


def generate_graph(xr, yr, obs_list):
    # Generate world graph with obstacles
    ws = [[0] * yr for i in range(xr)]
    for obs in obs_list:
        for i in range(obs[0][0], obs[0][1]):
            for j in range(obs[1][0], obs[1][1]):
                ws[i][j] = 1
    return ws


def show_result(op_path, world_state, robot_pose, goal_pose, obs_list):
    # Plot to show result if we found a path
    OPX = []
    OPY = []
    SPX = []
    SPY = []
    WX = []
    WY = []
    SX = robot_pose[0]
    SY = robot_pose[1]
    GX = goal_pose[0]
    GY = goal_pose[1]

    if op_path:
        for node in op_path:
            OPX.append(node[0])
            OPY.append(node[1])

    xr = len(world_state)
    yr = len(world_state[0])
    plt.figure(figsize=(8 * (xr // yr), 8))
    plt.plot(SX, SY, "bo", label="Robot_Pose", markersize=8)
    plt.plot(GX, GY, "ro", label="Goal_Pose", markersize=8)

    for obs in obs_list:
        OX = []
        OY = []
        if obs[0][1] - obs[0][0] == 1:
            for i in range(obs[1][0], obs[1][1]):
                OX.append(obs[0][0])
                OY.append(i)
        else:
            for i in range(obs[0][0], obs[0][1]):
                OX.append(i)
                OY.append(obs[1][0])
        plt.plot(OX, OY, "k", linewidth=5)

    if op_path:
        plt.plot(OPX, OPY, alpha=0.9, label="Optimal_Path", linewidth=3)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
    plt.show()
    plt.close()

class Astar(object):
    def __init__(self):
        self.x_range = 50
        self.y_range = 50
        self.x_resolution = 10 / self.x_range
        self.y_resolution = 10 / self.y_range
        self.robot_size = 0
        self.ws = [[0] * self.y_range for i in range(self.x_range)]

    def set_state(self, state):
        obs_list = generate_obstacle(self.x_range, self.y_range)
        world_state = generate_graph(self.x_range, self.y_range, obs_list)
        robot_pose, goal_pose = (5, 25), (self.x_range - 5, 25)
        # Run optimal planner
        search = Search(world_state, robot_pose, goal_pose, obs_list, self.robot_size)
        optimal_path = search.A_star()
        if optimal_path:
            print("Optimal search succeed!")
        else:
            print("No optimal path is found!")
        show_result(optimal_path, world_state, robot_pose, goal_pose, obs_list)

    def show_result(self, op_path, world_state, robot_pose, goal_pose, cir_obs_list, line_obs_list):
        # Plot to show result if we found a path
        OPX = []
        OPY = []
        SPX = []
        SPY = []
        WX = []
        WY = []
        SX = robot_pose[0]
        SY = robot_pose[1]
        GX = goal_pose[0]
        GY = goal_pose[1]

        if op_path:
            for node in op_path:
                OPX.append(node[0])
                OPY.append(node[1])

        xr = len(world_state)
        yr = len(world_state[0])
        plt.figure(figsize=(8 * (xr // yr), 8))
        plt.plot(SX, SY, "bo", label="Robot_Pose", markersize=8)
        plt.plot(GX, GY, "ro", label="Goal_Pose", markersize=8)

        for obs in cir_obs_list:
            OX = []
            OY = []
            for i in range(-obs[2], obs[2]+1):
                if i + obs[0] >=0 and i + obs[0] <self.x_range:
                    for j in range(-obs[2], obs[2]+1):
                        if j + obs[1] >= 0 and j + obs[1] < self.y_range:
                            if i*i + j*j < obs[2] * obs[2]:
                                OX.append(obs[0]+i)
                                OY.append(obs[1]+j)
            plt.plot(OX, OY, ".", linewidth=2)

        if op_path:
            plt.plot(OPX, OPY, alpha=0.9, label="Optimal_Path", linewidth=3)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
        plt.show()
        plt.close()

    def generate_obstacle2(self, human_state, obstacle_state, wall_state):
        cir_obs_list = []
        line_obs_list = []
        for human in human_state + obstacle_state:
            # radius = human.radius + 0.3
            # # human_size = np.ceil(radius / self.x_resolution)
            # # human_size = human_size.astype(np.int16)
            # human_pos = np.array([human.px, human.py])
            # # human_pos = self.transform(human_pos)
            # cir_obs_list.append(np.array([human_pos[0], human_pos[1],human_size]))

            cir_obs_list.append(np.array([human.px, human.py, human.radius + 0.4]))

        for wall in wall_state:
            # radius = 0.3
            # wall_size = np.ceil(radius/self.x_resolution)
            # wall_size= wall_size.astype(np.int16)
            # wall_start = self.transform(np.array([wall.sx, wall.sy]))
            # wall_end = self.transform(np.array([wall.ex, wall.ey]))
            # line_obs_list.append(np.array([wall_start[0], wall_start[1], wall_end[0], wall_end[1],wall_size]))
            line_obs_list.append(np.array([wall.sx,wall.sy,wall.ex,wall.ey,0.4]))
        return cir_obs_list, line_obs_list


    def transform(self, pos):
        x = np.ceil((pos[0] + 5.0) / self.x_resolution)
        y = np.ceil((pos[1] + 5.0) / self.y_resolution)
        return np.array([x,y], dtype=int)

    def inverse(self, coordinate):
        x = coordinate[0] * self.x_resolution - 5.0
        y = coordinate[1] * self.y_resolution - 5.0
        return (x,y)

    def set_state2(self, state):
        robot_state = state[0]
        human_state = state[1]
        obstacle_state = state[2]
        wall_state = state[3]
        circular_list, line_list = self.generate_obstacle2(human_state, obstacle_state, wall_state)
        robot_pos = self.transform(np.array([robot_state.px, robot_state.py]))
        goal_pos = self.transform(np.array([robot_state.gx, robot_state.gy]))
        robot_pose, goal_pose = (robot_pos[0], robot_pos[1]), (goal_pos[0], goal_pos[1])
        # # Run optimal planner
        search = Search2(self.ws, robot_pose, goal_pose, circular_list, line_list, self.robot_size)
        optimal_path = search.A_star()
        # self.show_result(optimal_path, self.ws, robot_pose, goal_pose, circular_list, line_list)
        if optimal_path is None:
            return None
        else:
            if len(optimal_path) > 5:
                return self.inverse(optimal_path[2])
            else:
                return None

def main():
    # Parameter initialization
    state=None
    astar_planner = Astar()
    astar_planner.set_state(state)

if __name__ == '__main__':
    main()