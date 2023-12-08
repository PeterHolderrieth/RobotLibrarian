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
# Own utils
from hwstation.utils import init_diagram, fix_input_port, visualize_diagram
from hwstation.add_objects import get_library_scenario_data, get_library_scenario_data_without_robot

from enum import Enum
import pandas as pd

def get_table_pointclouds(diagram_context: Context, diagram: Diagram):
    point_cloud_dict = {}
    for idx in range(4):
        point_cloud_dict[f"table_camera_{idx}_ptcloud"] = diagram.GetOutputPort(f"table_camera_{idx}_ptcloud").Eval(diagram_context)
    return point_cloud_dict


def merge_point_clouds(table_pointclouds: dict, 
                        downsample_factor: float = 0.005,
                        lower_xyz: List[float] = [0.0, -0.2, 0.5564], 
                        upper_xyz: List[float] = [0.75, 1.5, 0.8]):
    pcd = []
    for key in table_pointclouds.keys():
        cloud = table_pointclouds[key]
        pcd.append(
            cloud.Crop(lower_xyz=lower_xyz, upper_xyz=upper_xyz)
            )
    merged_pcd = Concatenate(pcd)
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
    return down_sampled_pcd

def get_merged_pointcloud(diagram_context: Context, diagram: Diagram):
    
    #Get merged point cloud from all cameras:
    table_pointclouds = get_table_pointclouds(diagram_context, diagram)
    merged_pcd = merge_point_clouds(table_pointclouds)

    #Ensure that all number are finite:
    merged_pcd_np = merged_pcd.xyzs().transpose()
    mask_points = (merged_pcd_np== np.inf).all(axis=1)
    if mask_points.any():
        sys.exit("No infinite points were expected")
    return merged_pcd

def convert_obj_to_pc(filename: str, n_samples: int = 10000, show: bool =False) -> np.ndarray:
    book_mesh = trimesh.load(filename)
    book_hull = book_mesh.convex_hull
    sample_points = book_hull.sample(n_samples)
    point_cloud = trimesh.points.PointCloud(sample_points)
    if show:
        scene = trimesh.Scene([book_hull, point_cloud])
        scene.show()
    return np.array(point_cloud.vertices).transpose()

def run_table_book_icp(diagram, diagram_context, meshcat):
    
    # Point clouds to obtain the cloud we will work with for grasps:
    scene_pcl = get_merged_pointcloud(diagram_context, diagram)
    meshcat.SetObject("merged_cropped_pcl", cloud=scene_pcl, point_size=0.004)

    book_filename = "hwstation/objects/book.obj"
    model_pcl = convert_obj_to_pc(book_filename, show=False)

    height_table = 0.5
    initial_guess = RigidTransform(p=[0.0,0.0,height_table])

    X_MS_hat, chat = IterativeClosestPoint(
        p_Om=model_pcl,
        p_Ws=scene_pcl.xyzs(),
        X_Ohat=initial_guess,
        meshcat=meshcat,
        meshcat_scene_path="icp",
        max_iterations=45,
    )

    transformed_model_pcl = X_MS_hat @ model_pcl

    cloud = PointCloud(transformed_model_pcl.shape[1])
    cloud.mutable_xyzs()[:] = transformed_model_pcl
    cloud.EstimateNormals(radius=0.5, num_closest=50)

    return cloud, X_MS_hat

def check_quality_of_icp(diagram: Diagram, diagram_context: Context):
    plant = diagram.GetSubsystemByName("station").GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    book_frame = plant.GetFrameByName("book")
    X_book = book_frame.CalcPoseInWorld(plant_context)
    Rot_est = X_MS_hat.rotation().matrix().flatten()
    trans_est = X_MS_hat.translation().flatten()
    Rot_truth = X_book.rotation().matrix().flatten()
    trans_truth = X_book.translation().flatten()
    est = np.concatenate([Rot_est,trans_est])
    truth = np.concatenate([Rot_truth,trans_truth])
    pd.DataFrame({"estimated": est, "truth": truth}).plot(kind='bar')