from IPython.display import HTML, SVG, display
from pydrake.systems.framework import Diagram
from manipulation.station import MakeHardwareStation, load_scenario
from manipulation.utils import ConfigureParser
from pydrake.all import (
    RigidTransform,
    Simulator,
    MultibodyPlant,
    DiagramBuilder,
    Meshcat, 
    DiagramBuilder, 
    Parser, 
    AddMultibodyPlantSceneGraph,
    MeshcatVisualizer,
    Diagram
)

from manipulation.station import AddPointClouds

import pydot

def setup_builder(meshcat: Meshcat, scenario_data: str):

    meshcat.Delete()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.00)
    
    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    parser = Parser(plant)
    ConfigureParser(parser)

    # Adding point cloud extractors:
    to_point_cloud = AddPointClouds(
        scenario=scenario, station=station, builder=builder, meshcat=meshcat
    )

    #Connect point clouds with output port:
    for idx, name in enumerate(to_point_cloud.keys()):
        builder.ExportOutput(
            to_point_cloud[name].get_output_port(), name+"_ptcloud"
        )

    
    return builder, plant, scene_graph, station, parser, scenario

def plot_and_simulate(meshcat: Meshcat, builder: DiagramBuilder, plant: MultibodyPlant, station: Diagram, time_end: float):
    plant.Finalize()
    visualizer = MeshcatVisualizer.AddToBuilder(builder, station.GetOutputPort("query_object"), meshcat)
    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)
    simulator = Simulator(diagram)
    plant_context = plant.GetMyContextFromRoot(simulator.get_mutable_context())
    simulator.AdvanceTo(time_end)
    return diagram, plant_context, simulator
    
def visualize_diagram(diagram: Diagram):
    return SVG(pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].create_svg())