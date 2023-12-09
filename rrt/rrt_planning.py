import numpy as np
import trimesh
import sys
from typing import List 
from IPython.display import clear_output
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PointCloud,
    RigidTransform,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
    Context,
    Diagram,
    PointCloud,
    Simulator,
    TrajectorySource,
    Solve,
    RotationMatrix,
    MultibodyPlant,
    eq,
    le,
    ge,
    StateInterpolatorWithDiscreteDerivative,
    MinimumDistanceLowerBoundConstraint,
    RollPitchYaw,
    SolutionResult,
    CollisionFilterDeclaration,
    GeometrySet,
    Role,
)
from pydrake.geometry import Meshcat
from pydrake.multibody import inverse_kinematics

from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation import running_as_notebook
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb, AddMultibodyTriad, MakeManipulationStation
from manipulation.utils import ConfigureParser
from manipulation.clutter import GraspCandidateCost, GenerateAntipodalGraspCandidate
from manipulation.icp import IterativeClosestPoint
from manipulation.pick import (
    MakeGripperCommandTrajectory,
    MakeGripperFrames,
    MakeGripperPoseTrajectory,
)
# Own utils
from hwstation.utils import init_diagram, fix_input_port, visualize_diagram
from hwstation.add_objects import get_library_scenario_data, get_library_scenario_data_without_robot

from enum import Enum
import pandas as pd
from random import random

from rrt.rrt_planner_helpers.rrt_planning import (
    Problem,
)
from rrt.rrt_planner_helpers.robot import (
    ConfigurationSpace,
    Range,
)
from typing import Tuple, List, Dict

class IKSolver(object):
    def __init__(self):

        # setup controller plant
        plant_mobile_iiwa = MultibodyPlant(0.0)
        iiwa_file = "file:///workspaces/RobotLibrarian/hwstation/objects/mobile_iiwa14_primitive_collision.urdf"
        iiwa = Parser(plant_mobile_iiwa).AddModelsFromUrl(iiwa_file)[0]

        # Define frames
        L0 = plant_mobile_iiwa.GetFrameByName("base")
        l7_frame = plant_mobile_iiwa.GetFrameByName("iiwa_link_7")
        world_frame = plant_mobile_iiwa.world_frame()
        plant_mobile_iiwa.WeldFrames(world_frame, L0)
        plant_mobile_iiwa.Finalize()
        plant_mobile_context = plant_mobile_iiwa.CreateDefaultContext()

        # gripper in link 7 frame
        X_L7G = RigidTransform(
            rpy=RollPitchYaw([np.pi / 2, 0, np.pi / 2]), p=[0, 0, 0.114]
        )
        world_frame = plant_mobile_iiwa.world_frame()

        self.world_frame = world_frame
        self.l7_frame = l7_frame
        self.plant_iiwa = plant_mobile_iiwa
        self.plant_context = plant_mobile_context
        self.X_L7G = X_L7G
        self.joint_limits = get_joint_limits(self.plant_iiwa)

    def solve(self, X_WT, q_guess=None, theta_bound=0.01, position_bound=0.01):
        """
        plant: a mini plant only consists of iiwa arm with no gripper attached
        X_WT: transform of target frame in world frame
        q_guess: a guess on the joint state sol
        """
        plant_mobile_iiwa = self.plant_iiwa
        l7_frame = self.l7_frame
        X_L7G = self.X_L7G
        world_frame = self.world_frame

        R_WT = X_WT.rotation()
        p_WT = X_WT.translation()

        if q_guess is None:
            q_guess = np.zeros(10)

        ik_instance = inverse_kinematics.InverseKinematics(plant_mobile_iiwa)
        # align frame A to frame B
        ik_instance.AddOrientationConstraint(
            frameAbar=l7_frame,
            R_AbarA=X_L7G.rotation(),
            #   R_AbarA=RotationMatrix(), # for link 7
            frameBbar=world_frame,
            R_BbarB=R_WT,
            theta_bound=position_bound,
        )

        # align point Q in frame B to the bounding box in frame A
        ik_instance.AddPositionConstraint(
            frameB=l7_frame,
            p_BQ=X_L7G.translation(),
            # p_BQ=[0,0,0], # for link 7
            frameA=world_frame,
            p_AQ_lower=p_WT - position_bound,
            p_AQ_upper=p_WT + position_bound,
        )
        prog = ik_instance.prog()
        prog.SetInitialGuess(ik_instance.q(), q_guess)
        
        prog.AddConstraint(le(ik_instance.q(),self.joint_limits[:,1]))
        prog.AddConstraint(ge(ik_instance.q(),self.joint_limits[:,0]))

        result = Solve(prog)
        if result.get_solution_result() != SolutionResult.kSolutionFound:
            return result.GetSolution(ik_instance.q()), False
        return result.GetSolution(ik_instance.q()), True

class ManipulationStationSim:
    def __init__(self, diagram: Diagram, diagram_context: Context):
        self.diagram = diagram
        self.station = diagram.GetSubsystemByName("station")
        self.plant = self.station.GetSubsystemByName("plant")
        self.scene_graph = self.station.GetSubsystemByName("scene_graph")

        # scene graph query output port.
        self.query_output_port = self.scene_graph.GetOutputPort("query")

        # contexts
        self.context_diagram = diagram_context
        self.context_station = self.diagram.GetSubsystemContext(
            self.station, self.context_diagram
        )
        self.context_scene_graph = self.station.GetSubsystemContext(
            self.scene_graph, self.context_station
        )
        self.context_plant = self.station.GetMutableSubsystemContext(
            self.plant, self.context_station
        )
        # # mark initial configuration
        # self.q0 = self.plant.GetPositions(
        #     self.context_plant, self.plant.GetModelInstanceByName("mobile_iiwa")
        # )

    def SetStationConfiguration(
        self, q_iiwa, gripper_setpoint
    ):
        """
        :param q_iiwa: (10,) numpy array, base pos and joint angle of robots in radian.
        :param gripper_setpoint: float, gripper opening distance in meters.
        :return:
        """
        self.plant.SetPositions(
            self.context_plant,
            self.plant.GetModelInstanceByName("mobile_iiwa"),
            q_iiwa,
        )
        self.plant.SetPositions(
            self.context_plant,
            self.plant.GetModelInstanceByName("wsg"),
            [-gripper_setpoint / 2, gripper_setpoint / 2],
        )

    def ExistsCollision(
        self, q_iiwa, gripper_setpoint):
        self.SetStationConfiguration(
            q_iiwa, gripper_setpoint
        )
        query_object = self.query_output_port.Eval(self.context_scene_graph)
        collision_pairs = query_object.ComputePointPairPenetration()
        return len(collision_pairs) > 0

def get_joint_limits(plant: MultibodyPlant,
        x_limits: Tuple[float,float] = (0.0,20.0),
        y_limits: Tuple[float,float] = (0.0,20.0),
        z_limits: Tuple[float,float] = (0.0,0.25),
        joint_limits_dict: Dict[str,Tuple[float,float]] = {}):

    nq = 10
    joint_limits = np.zeros((nq, 2))
    
    joint = plant.GetJointByName("iiwa_base_x")
    joint_limits[0, 0] = x_limits[0]
    joint_limits[0, 1] = x_limits[1]

    joint = plant.GetJointByName("iiwa_base_y")
    joint_limits[1, 0] = y_limits[0]
    joint_limits[1, 1] = y_limits[1]

    joint = plant.GetJointByName("iiwa_base_z")
    joint_limits[2, 0] = z_limits[0]
    joint_limits[2, 1] = z_limits[1]


    for i in range(4,nq):
        joint_name = "iiwa_joint_%i" % (i-2)
        joint = plant.GetJointByName("iiwa_joint_%i" % (i-2))
        if joint_limits_dict.get(joint_name,False):
            joint_limits[i, 0] = max(joint_limits_dict[joint_name][0],joint.position_lower_limits())
            joint_limits[i, 1] = min(joint_limits_dict[joint_name][1],joint.position_upper_limits())
        else:
            joint_limits[i, 0] = joint.position_lower_limits()
            joint_limits[i, 1] = joint.position_upper_limits()


    #Special case: Joint 1 (shouldn't be infinity)
    # joint_limits[3, 0] = max(-10, joint_limits[3, 0])
    # joint_limits[3, 1] = min(10, joint_limits[3, 1])

    joint_name = "iiwa_joint_1"
    if joint_limits_dict.get(joint_name,False):
        joint_limits[3, 0] = max(joint_limits_dict[joint_name][0],-10)
        joint_limits[3, 1] = min(joint_limits_dict[joint_name][1],10)
    else:
        joint_limits[3, 0] = -10
        joint_limits[3, 1] = 10

    return joint_limits

class IiwaProblem(Problem):
    def __init__(
        self,
        q_start: np.array,
        q_goal: np.array,
        gripper_setpoint: float,
        collision_checker: ManipulationStationSim,
        debug: bool = False,
        x_limits: Tuple[float,float] = (0.0,20.0),
        y_limits: Tuple[float,float] = (0.0,20.0),
        z_limits: Tuple[float,float] = (0.0,0.25),
        joint_limits_dict: Dict[str,Tuple[float,float]] = {},
    ):
        self.gripper_setpoint = gripper_setpoint
        self.collision_checker = collision_checker
 
        joint_limits = get_joint_limits(plant = self.collision_checker.plant,
        x_limits=x_limits,y_limits=y_limits,z_limits=z_limits,
        joint_limits_dict = joint_limits_dict)
        nq = 10

        range_list = []
        for joint_limit in joint_limits:
            range_list.append(Range(joint_limit[0], joint_limit[1]))
        #print(range_list)
        def l2_distance(q: tuple):
            sum = 0
            for q_i in q:
                sum += q_i**2
            return np.sqrt(sum)
        self.joint_limits = joint_limits
        max_steps = nq * [np.pi / 180 * 2]  # three degrees
        cspace_iiwa = ConfigurationSpace(range_list, l2_distance, max_steps)
        self.joint_limits = joint_limits

        #Call base class constructor.
        Problem.__init__(
            self,
            x=10,  # not used.
            y=10,  # not used.
            robot=None,  # not used.
            obstacles=None,  # not used.
            start=tuple(q_start),
            goal=tuple(q_goal),
            cspace=cspace_iiwa,
            debug=debug
        )

    def collide(self, configuration):
        q = np.array(configuration)
        return self.collision_checker.ExistsCollision(
            q,
            self.gripper_setpoint,
        )
    
class TreeNode:
    def __init__(self, value, parent=None):
        self.value = value  # tuple of floats representing a configuration
        self.parent = parent  # another TreeNode
        self.children = []  # list of TreeNodes

class RRT:
    """
    RRT Tree.
    """

    def __init__(self, root: TreeNode, cspace: ConfigurationSpace):
        self.root = root  # root TreeNode
        self.cspace = cspace  # robot.ConfigurationSpace
        self.size = 1  # int length of path
        self.max_recursion = 1000  # int length of longest possible path

    def add_configuration(self, parent_node, child_value):
        child_node = TreeNode(child_value, parent_node)
        parent_node.children.append(child_node)
        self.size += 1
        return child_node

    # Brute force nearest, handles general distance functions
    def nearest(self, configuration):
        """
        Finds the nearest node by distance to configuration in the
             configuration space.

        Args:
            configuration: tuple of floats representing a configuration of a
                robot

        Returns:
            closest: TreeNode. the closest node in the configuration space
                to configuration
            distance: float. distance from configuration to closest
        """

        configuration = np.nan_to_num(configuration) # had to add this line to make it work
        assert self.cspace.valid_configuration(configuration)

        def recur(node, depth=0):
            closest, distance = node, self.cspace.distance(
                node.value, configuration
            )
            if depth < self.max_recursion:
                for child in node.children:
                    (child_closest, child_distance) = recur(child, depth + 1)
                    if child_distance < distance:
                        closest = child_closest
                        child_distance = child_distance
            return closest, distance

        return recur(self.root)[0]
    
class RRT_tools:
    def __init__(self, problem):
        # rrt is a tree
        self.rrt_tree = RRT(TreeNode(problem.start), problem.cspace)
        problem.rrts = [self.rrt_tree]
        self.problem = problem

    def find_nearest_node_in_RRT_graph(self, q_sample):
        nearest_node = self.rrt_tree.nearest(q_sample)
        return nearest_node

    def sample_node_in_configuration_space(self):
        q_sample = self.problem.cspace.sample()
        return q_sample

    def calc_intermediate_qs_wo_collision(self, q_start, q_end):
        """create more samples by linear interpolation from q_start
        to q_end. Return all samples that are not in collision

        Example interpolated path:
        q_start, qa, qb, (Obstacle), qc , q_end
        returns >>> q_start, qa, qb
        """
        return self.problem.safe_path(q_start, q_end)

    def grow_rrt_tree(self, parent_node, q_sample):
        """
        add q_sample to the rrt tree as a child of the parent node
        returns the rrt tree node generated from q_sample
        """
        child_node = self.rrt_tree.add_configuration(parent_node, q_sample)
        return child_node

    def node_reaches_goal(self, node):
        return node.value == self.problem.goal

    def backup_path_from_node(self, node):
        path = [node.value]
        while node.parent is not None:
            node = node.parent
            path.append(node.value)
        path.reverse()
        return path

def rrt_planning(problem, max_iterations=1000, prob_sample_q_goal=0.05):
    """
    Input:
        problem (IiwaProblem): instance of a utility class
        max_iterations: the maximum number of samples to be collected
        prob_sample_q_goal: the probability of sampling q_goal

    Output:
        path (list): [q_start, ...., q_goal].
                    Note q's are configurations, not RRT nodes
    """
    rrt_tools = RRT_tools(problem)
    q_goal = problem.goal
    q_start = problem.start

    for k in range(max_iterations):
        q_sample = rrt_tools.sample_node_in_configuration_space()
        rand = random()
        if rand < prob_sample_q_goal:
            q_sample = q_goal
        nearest_node = rrt_tools.find_nearest_node_in_RRT_graph(q_sample)
        q_list = rrt_tools.calc_intermediate_qs_wo_collision(nearest_node.value, q_sample)

        last_node = nearest_node
        for q_i in q_list:
            last_node = rrt_tools.grow_rrt_tree(last_node, q_i)
        

        path_found = rrt_tools.node_reaches_goal(last_node)
        if hasattr(path_found, "__len__"):
            path_found = all(rrt_tools.node_reaches_goal(last_node))
        
        if path_found:
            path = rrt_tools.backup_path_from_node(last_node)
            return path

    return None