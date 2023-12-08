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

from enum import Enum
import pandas as pd

def make_internal_model():
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)

    parser.AddModelsFromUrl("file:///workspaces/RobotLibrarian/hwstation/objects/library_setup_floating_gripper.dmd.yaml")
    plant.Finalize()

    return builder.Build()

def sample_grasps(cloud: PointCloud, diagram: Diagram, diagram_context: Context):
    rng = np.random.default_rng()
    # Now find grasp poses
    # X_Gs will have the poses to be used for planning when working on that step
    internal_model = make_internal_model()
    internal_model_context = internal_model.CreateDefaultContext()
    costs = []
    X_Gs = []
    for i in range(100 if running_as_notebook else 2):
        cost, X_G = GenerateAntipodalGraspCandidate(
            internal_model, internal_model_context, cloud, rng
        )
        if np.isfinite(cost):
            costs.append(cost)
            X_Gs.append(X_G)

    indices = np.asarray(costs).argsort()[:5]
    min_cost_XGs = []
    for idx in indices:
        min_cost_XGs.append(X_Gs[idx])

    #Get the antipodal grasp that is closest to the robot WSG frame:
    positions = np.stack([frame.translation() for frame in min_cost_XGs])

    plant = diagram.GetSubsystemByName("station").GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(diagram_context)

    wsg = plant.GetBodyByName("body")
    wsg_body_index = wsg.index()

    wsg_pose = plant.EvalBodyPoseInWorld(plant_context,wsg)
    wsg_position = wsg_pose.translation()

    best_grasp_idx = np.argmin(((positions-wsg_position)**2).sum(axis=1))
    X_G_optim = min_cost_XGs[best_grasp_idx]

    return X_G_optim