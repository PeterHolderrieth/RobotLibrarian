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

    
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

def get_random_rotation_matrix():
    return RotationMatrix(RigidTransform(R=RotationMatrix.MakeZRotation(theta=np.random.uniform(0,2*np.pi))).multiply(
    RigidTransform(R=RotationMatrix.MakeXRotation(theta=np.random.uniform(0,2*np.pi)))).multiply(
    RigidTransform(R=RotationMatrix.MakeYRotation(theta=np.random.uniform(0,2*np.pi)))).rotation().matrix())

def cluster_merged_pointcloud(pcloud_arr: np.ndarray, sort_x_comp: bool = True, visualize: bool = False, meshcat: Meshcat = None):
    adj = kneighbors_graph(pcloud_arr, 5, mode='distance', include_self=True, metric='euclidean')
    n_components, labels = csgraph.connected_components(adj)
    pcloud_label_list = []
    cluster_mean_list = []
    for idx in range(n_components):
        mask_pc = (labels==idx)
        cloud = PointCloud(mask_pc.sum())
        cluster_mean = pcloud_arr[mask_pc].mean(axis=0)
        cluster_mean_list.append(cluster_mean)
        cloud.mutable_xyzs()[:] =  pcloud_arr[mask_pc].transpose()
        pcloud_label_list.append(cloud)
    
    if sort_x_comp:
        sort_x_comp = np.argsort([cmean[0] for cmean in cluster_mean_list])
        cluster_mean_list = np.array(cluster_mean_list)[sort_x_comp].tolist()
        pcloud_label_list = np.array(pcloud_label_list)[sort_x_comp].tolist()

    if visualize:
        for idx in range(n_components):
            meshcat.SetObject(f"clusterd_pcl_{idx}", cloud=pcloud_label_list[idx], point_size=0.004)
            
    return pcloud_label_list, cluster_mean_list


N_CAMERAS = 10

def get_table_pointclouds(diagram_context: Context, diagram: Diagram):
    point_cloud_dict = {}
    for idx in range(N_CAMERAS):
        point_cloud_dict[f"table_camera_{idx}_ptcloud"] = diagram.GetOutputPort(f"table_camera_{idx}_ptcloud").Eval(diagram_context)
    return point_cloud_dict


def merge_point_clouds(table_pointclouds: dict, 
                        downsample_factor: float = 0.005,
                        lower_xyz: List[float] = [0.25, 0.0, 0.56], 
                        upper_xyz: List[float] = [1.75, 0.5, 0.8]):
    pcd = []
    for key in table_pointclouds.keys():
        cloud = table_pointclouds[key]
        pcd.append(
            cloud.Crop(lower_xyz=lower_xyz, upper_xyz=upper_xyz)
            )
    merged_pcd = Concatenate(pcd)
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=downsample_factor)
    return down_sampled_pcd

def get_merged_pointcloud(diagram_context: Context, diagram: Diagram,
                          downsample_factor: float = 0.005,
                          lower_xyz: List[float] = [0.25, 0.0, 0.56], 
                          upper_xyz: List[float] = [1.75, 0.5, 0.8]):
    
    #Get merged point cloud from all cameras:
    table_pointclouds = get_table_pointclouds(diagram_context, diagram)
    merged_pcd = merge_point_clouds(table_pointclouds,downsample_factor=downsample_factor, lower_xyz=lower_xyz, upper_xyz=upper_xyz)

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

def run_table_multi_book_icp(diagram, diagram_context, meshcat, visualize: bool = False, icp_limit: int = N_CAMERAS+10, align_frames: bool = True):

    # Point clouds to obtain the cloud we will work with for grasps:
    merged_pointcloud = get_merged_pointcloud(diagram_context, diagram)
    n_points = merged_pointcloud.xyzs().shape[1]
    
    #If there are less than 10 points on the table, there are no books:
    if n_points < 10:
        return [],[]
    
    if visualize:
        meshcat.SetObject("merged_cropped_pcl", cloud=merged_pointcloud, point_size=0.004)

    book_filename = "hwstation/objects/book.obj"
    model_pcl = convert_obj_to_pc(book_filename, show=False)
    
    pcloud_arr = merged_pointcloud.xyzs().transpose()
    pcloud_label_list, cluster_mean_list = cluster_merged_pointcloud(pcloud_arr,visualize=False, meshcat=meshcat)

    X_MS_hat_list = []
    cloud_list = []
    for idx, scene_pcl in enumerate(pcloud_label_list):
        if align_frames:
            initial_guess = RigidTransform(p=[0,cluster_mean_list[idx][1],cluster_mean_list[idx][2]])
        else:
            height_table = 0.5
            initial_guess = RigidTransform(p=[0,0,height_table])

        while True:
            X_MS_hat, chat = IterativeClosestPoint(
                p_Om=model_pcl,
                p_Ws=scene_pcl.xyzs(),
                X_Ohat=initial_guess,
                meshcat=meshcat,
                meshcat_scene_path=f"icp_{idx}",
                max_iterations=100,
            )
            
            y_crit = np.dot(X_MS_hat.rotation().matrix()[:,1],np.array([1.0,0.0,0.0])) > 0.9

            if y_crit or (not align_frames):

                transformed_model_pcl = X_MS_hat @ model_pcl
                cloud = PointCloud(transformed_model_pcl.shape[1])
                cloud.mutable_xyzs()[:] = transformed_model_pcl
                cloud.EstimateNormals(radius=0.5, num_closest=50)
                cloud_list.append(cloud)
                X_MS_hat_list.append(X_MS_hat)
                break
            else:
                initial_guess = RigidTransform(RotationMatrix.MakeZRotation(np.pi)).multiply(X_MS_hat)
                print(f"Retrying fitting idx={idx}")

        #If we only search for the top-icp_limit pointclouds, then stop here:
        if idx >= icp_limit:
            break

    return cloud_list, X_MS_hat_list

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