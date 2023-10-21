from IPython.display import HTML, SVG, display
from pydrake.systems.framework import Diagram
import pydot

def visualize_diagram(diagram: Diagram):
    return SVG(pydot.graph_from_dot_data(diagram.GetGraphvizString())[0].create_svg())
