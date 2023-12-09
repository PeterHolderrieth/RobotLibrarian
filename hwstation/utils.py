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
    MultibodyPlant,
    CollisionFilterDeclaration,
    GeometrySet,
    Role,
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

def init_builder(meshcat: Meshcat, scenario_data: str):
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
    
    builder.ExportInput(
        station.GetInputPort("wsg.position"),
        "wsg.position"
    )
    builder.ExportInput(
        station.GetInputPort("wsg.force_limit"),
        "wsg.force_limit"
    )

    builder.ExportOutput(
        station.GetOutputPort("wsg.state_measured"),
        "wsg.state_measured"
    )
    builder.ExportOutput(
        station.GetOutputPort("wsg.force_measured"),
        "wsg.force_measured"
    )

    visualizer = MeshcatVisualizer.AddToBuilder(builder, station.GetOutputPort("query_object"), meshcat)
    print("Station input port size: ", station.GetInputPort("mobile_iiwa.desired_state"))
    plant = station.GetSubsystemByName("plant")
    print("plant.GetStateNames(): ", len(plant.GetStateNames()))
    print("plant.GetActuatorNames(): ", len(plant.GetActuatorNames()))
    return builder, visualizer, station

def init_diagram(meshcat: Meshcat, scenario_data: str):
    builder, visualizer, station = init_builder(meshcat, scenario_data)
    diagram = builder.Build()
    #diagram.set_name("plant and scene_graph")
    simulator = Simulator(diagram)
    return diagram, visualizer, simulator

def fix_input_port(diagram: Diagram, simulator: Simulator):
    
    sim_context = simulator.get_mutable_context()

    x0 = diagram.GetOutputPort("mobile_iiwa.state_estimated").Eval(sim_context)
    x0[2] = 0.001
    print("Fixing input port of size: ", len(x0))
    diagram.GetInputPort("mobile_iiwa.desired_state").FixValue(sim_context, x0)

    wsgx0 = [0.1] #diagram.GetOutputPort("wsg.state_measured").Eval(sim_context)[0]]
    print("Fixing input port of size: ", len(wsgx0))
    diagram.GetInputPort("wsg.position").FixValue(sim_context, wsgx0)

def visualize_diagram(diagram: Diagram):
    return SVG(pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].create_svg())

def filterCollsionGeometry(scene_graph, context=None):
    """Some robot models may appear to have self collisions due to overlapping collision geometries.
    This function filters out such problems for our PR2 model."""
    if context is None:
        filter_manager = scene_graph.collision_filter_manager()
    else:
        filter_manager = scene_graph.collision_filter_manager(context)
    inspector = scene_graph.model_inspector()

    book = {}
    shelf = {}
    table = {}

    for gid in inspector.GetGeometryIds(
        GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity
    ):
        gid_name = inspector.GetName(inspector.GetFrameId(gid))

        if "book" in gid_name:
            link_name = gid_name.split("::")[1]
            book[link_name] = [gid]

        if "table" in gid_name:
            link_name = gid_name.split("::")[1]
            table[link_name] = [gid]

        if "shelf" in gid_name:
            link_name = gid_name.split("::")[1]
            shelf[link_name] = [gid]

    def add_exclusion(set1, set2=None):
        if set2 is None:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(GeometrySet(set1))
            )
        else:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    GeometrySet(set1), GeometrySet(set2)
                )
            )

    # Exclude collisions between book and table
    for _, book_ids in book.items():
        for _, table_piece_ids in table.items():
            add_exclusion(
                table_piece_ids, book_ids
            )

    # Exclude collisions between book and shelf
    for _, book_ids in book.items():
        for _, shelf_piece_ids in shelf.items():
            add_exclusion(
                shelf_piece_ids, book_ids
            )

def get_iiwa_joint_state(diagram: Diagram, diagram_context: Context):
    plant = diagram.GetSubsystemByName("station").GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    q0 = plant.GetPositions(
                plant_context, plant.GetModelInstanceByName("mobile_iiwa")
            )
    return q0

def visualize_frame(name: str, meshcat: Meshcat, X_trans: RigidTransform, length=0.08, radius=0.006):
    AddMeshcatTriad(
    meshcat, f"painter/{name}", length=length, radius=radius, X_PT=X_trans,
)