import numpy as np
import trimesh
import sys
from typing import List 
from IPython.display import clear_output, SVG
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
    MultibodyPlant
)
from pydrake.geometry import Meshcat
from pydrake.multibody import inverse_kinematics

from manipulation.station import MakeHardwareStation, load_scenario
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation import running_as_notebook
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb, AddMultibodyTriad, MakeManipulationStation
from manipulation.utils import ConfigureParser
from manipulation.clutter import GraspCandidateCost, GenerateAntipodalGraspCandidate
from manipulation.icp import IterativeClosestPoint
from manipulation.station import AddPointClouds

# Own utils
from enum import Enum
import pydot

def init_diagram(meshcat: Meshcat, scenario_data: str):
    meshcat.Delete()
    builder = DiagramBuilder()
    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat,parser_preload_callback=ConfigureParser))

    # Adding point cloud extractors:
    to_point_cloud = AddPointClouds(
        scenario=scenario, station=station, builder=builder, meshcat=meshcat
    )

    #Connect point clouds with output port:
    for idx, name in enumerate(to_point_cloud.keys()):
        builder.ExportOutput(
            to_point_cloud[name].get_output_port(), name+"_ptcloud")

    builder.ExportOutput(
        station.GetOutputPort("mobile_iiwa.state_estimated"),
        "mobile_iiwa.state_estimated"
    )

    builder.ExportInput(
        station.GetInputPort("mobile_iiwa.desired_state"),
        "mobile_iiwa.desired_state"
    )

    visualizer = MeshcatVisualizer.AddToBuilder(builder, station.GetOutputPort("query_object"), meshcat)
    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")
    simulator = Simulator(diagram)
    return diagram, visualizer, simulator

def fix_input_port(diagram: Diagram, simulator: Simulator):
    sim_context = simulator.get_mutable_context()
    x0 = diagram.GetOutputPort("mobile_iiwa.state_estimated").Eval(sim_context)
    diagram.GetInputPort("mobile_iiwa.desired_state").FixValue(sim_context, x0)

def visualize_diagram(diagram: Diagram):
    return SVG(pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].create_svg())