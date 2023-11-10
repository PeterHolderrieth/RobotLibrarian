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

import pydot

def setup_builder(meshcat: Meshcat, scenario_data: str):

    meshcat.Delete()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.00)
    
    scenario = load_scenario(data=scenario_data)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))

    parser = Parser(plant)
    ConfigureParser(parser)
    
    return builder, plant, scene_graph, station, parser

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