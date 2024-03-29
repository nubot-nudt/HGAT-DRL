import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, JointState_2tyeps


class NHORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = 'NHORCA'
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 10
        self.max_neighbors = 10
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.radius = 0.3
        self.max_speed = 1
        self.sim = None
        self.speed_samples = 5
        self.rotation_samples = 16
        self.sampling = None
        self.rotation_constraint = None

    def configure(self, config, device='cpu'):
        self.set_common_parameters(config)
        return

    def set_common_parameters(self, config):
        self.kinematics = config.action_space.kinematics
        self.sampling = config.action_space.sampling
        self.speed_samples = config.action_space.speed_samples
        self.rotation_samples = config.action_space.rotation_samples
        self.rotation_constraint = config.action_space.rotation_constraint

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        state = self.state_transform(state)
        robot_state = state.robot_state
        cur_theta = robot_state.theta
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            robot__no = self.sim.addAgent(robot_state.position, *params, robot_state.radius, # + self.safety_space,
                              robot_state.v_pref, robot_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, human_state.radius + self.safety_space,
                                  self.max_speed, human_state.velocity)
        else:
            self.sim.setAgentPosition(0, robot_state.position)
            self.sim.setAgentVelocity(0, robot_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((robot_state.gx - robot_state.px, robot_state.gy - robot_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        speed_samples = self.speed_samples
        rotation_samples = self.rotation_samples
        d_vel = 1.0 / speed_samples / 2
        if self.kinematics is 'holonomic':
            action = ActionXY(*self.sim.getAgentVelocity(0))
        elif self.kinematics is 'unicycle':
            action = ActionXY(*self.sim.getAgentVelocity(0))
            theta = np.arctan2(action.vy, action.vx)
            vel = np.sqrt(action.vx*action.vx + action.vy * action.vy)
            theta = (theta - cur_theta + np.pi) % (2 * np.pi) - np.pi
            action = ActionRot(vel, theta)
        # vel_index = (np.sqrt(action.vx * action.vx + action.vy * action.vy) // d_vel + 1) // 2
        # if vel_index > speed_samples:
        #     vel_index = speed_samples
        # d_rot = np.pi / rotation_samples
        # rot_index = (np.arctan2(action.vy, action.vx) // d_rot + 1) // 2
        # if rot_index < 0:
        #     rot_index = rot_index + rotation_samples
        # if vel_index == 0:
        #     action_index = int(0)
        # else:
        #     action_index = int((vel_index - 1) * rotation_samples + rot_index + 1)
        # action = ActionXY(vel_index * d_vel * 2.0 * np.cos(rot_index * d_rot * 2.0), vel_index * d_vel * 2.0 *
        #                   np.sin(rot_index * d_rot * 2.0))
        action_index = -1
        self.last_state = state

        return action, action_index

    def state_transform(self,state):
        human_state = state.human_states
        obstacle_state = state.obstacle_states
        robot_state = state.robot_state
        for i in range((len(obstacle_state))):
            obstacle_human = ObservableState(obstacle_state[i].px, obstacle_state[i].py, 0.0, 0.0, obstacle_state[i].radius)

            human_state.append(obstacle_human)
        state = JointState_2tyeps(robot_state, human_state)
        return state