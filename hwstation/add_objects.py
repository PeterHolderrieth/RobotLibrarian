import numpy as np

ROW_DIST_SHELF = 2.5 #5.0 #1.7
COLUMN_DIST_SHELF = 1.7
SHELF_OFFSET_X = 1.7
SHELF_OFFSET_Y = 2.4
SHELF_OFFSET_Z = 0.42
N_ROWS = 3
N_COLUMNS = 2
SHELF_NAMES = "ABCDEFGHIJKLMNOPQRST"
TABLE_OFFSET = [1.0,0.25,0.53]
N_BOOKS = 6 #10
BOOK_DIST = 0.20 #0.13 

def get_empty_scenario_data() -> str:
    scenario_data = "directives:"
    return scenario_data

def add_ground_floor(scenario_data) -> str:
    scenario_data += """
- add_model:
    name: ground_floor
    file: file:///workspaces/RobotLibrarian/hwstation/objects/ground_floor.sdf
- add_weld:
    parent: world
    child: ground_floor::ground_floor
    X_PC:
        translation: [0.0,0.0,0.0]
"""
    return scenario_data

def add_shelves(scenario_data) -> str:

    shelf_name_idx = -1
    
    #Loop over all rows and columns and add a shelf
    for n_row in range(N_ROWS):
        for n_col in range(N_COLUMNS):
            
            #Get next shelf name:
            shelf_name_idx += 1
            shelf_name = SHELF_NAMES[shelf_name_idx]
    
            translation = np.array([SHELF_OFFSET_X,SHELF_OFFSET_Y,SHELF_OFFSET_Z]) \
                            + np.array([n_row * ROW_DIST_SHELF, n_col * COLUMN_DIST_SHELF, 0.0])

            scenario_data += """
- add_model:
    name: shelf_"""+str(shelf_name)+"""
    file: file:///workspaces/RobotLibrarian/hwstation/objects/library_shelf.sdf
- add_weld:
    parent: world
    child: shelf_"""+str(shelf_name)+"""::shelves_body
    X_PC:
        translation: ["""+str(translation[0])+""", """+str(translation[1])+""", """+str(translation[2])+"""]
"""
            
    return scenario_data


def add_table(scenario_data) -> str:

    scenario_data += """
- add_model:
    name: table
    file: file:///workspaces/RobotLibrarian/hwstation/objects/library_table_hydro.sdf
- add_weld:
    parent: world
    child: table
    X_PC:
        translation: ["""+str(TABLE_OFFSET[0])+""", """+str(TABLE_OFFSET[1])+""", """+str(TABLE_OFFSET[2])+"""]
"""
    return scenario_data

def add_books(scenario_data) -> str:

    book_offset = TABLE_OFFSET + np.array([-0.54,0.0,0.11])
    book_names = SHELF_NAMES

    for idx in range(N_BOOKS):
        book_translation = book_offset + np.array([idx*BOOK_DIST,0.0,0.0])
        scenario_data += """
- add_model:
    name: book_"""+str(book_names[idx])+"""
    file: file:///workspaces/RobotLibrarian/hwstation/objects/book.sdf
    default_free_body_pose:
        book:
            translation: ["""+str(book_translation[0])+""", """+str(book_translation[1])+""", """+str(book_translation[2])+"""]
            rotation: !Rpy { deg: [0, 0, -90]}
"""
# - add_weld:
#     parent: world
#     child: book_"""+str(book_names[idx])+"""::book
#     X_PC:
#         translation: ["""+str(book_translation[0])+""", """+str(book_translation[1])+""", """+str(book_translation[2])+"""]
#         rotation: !Rpy { deg: [0, 0, -90]}
    return scenario_data

#Note the camera frames are as follows:
#roll-pitch-yaw (Rpy) [0,0,0] corresponds
#to the camera facing straigh up in positive z-direction
def add_camera_visual(scenario_data) -> str:
    scenario_data +="""
- add_frame:
    name: camera_table_above
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [180.0, 0.0, 0.0]}
        translation: [0., 0., 0.7]

- add_model:
    name: camera0
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_table_above
    child: camera0::base

- add_frame:
    name: camera_table_left
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [-100.0, 0.0, 90.0]}
        translation: [0.9, 0., 0.3]

- add_model:
    name: camera1
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_table_left
    child: camera1::base

- add_frame:
    name: camera_table_right
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [-100.0, 0.0, -90.0]}
        translation: [-0.9, 0., 0.3]

- add_model:
    name: camera2
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_table_right
    child: camera2::base

- add_frame:
    name: camera_behind_table_1
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [-100.0, 0.0, 0.0]}
        translation: [0.5, -0.45, 0.3]

- add_model:
    name: camera3
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_behind_table_1
    child: camera3::base

- add_frame:
    name: camera_behind_table_2
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [-100.0, 0.0, 0.0]}
        translation: [0.25, -0.45, 0.3]

- add_model:
    name: camera4
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_behind_table_2
    child: camera4::base

- add_frame:
    name: camera_behind_table_3
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [-100.0, 0.0, 0.0]}
        translation: [0.0, -0.45, 0.3]

- add_model:
    name: camera5
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_behind_table_3
    child: camera5::base

- add_frame:
    name: camera_behind_table_4
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [-100.0, 0.0, 0.0]}
        translation: [-0.25, -0.45, 0.3]

- add_model:
    name: camera6
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_behind_table_4
    child: camera6::base

- add_frame:
    name: camera_behind_table_5
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [-100.0, 0.0, 0.0]}
        translation: [-0.5, -0.45, 0.3]

- add_model:
    name: camera7
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_behind_table_5
    child: camera7::base

- add_frame:
    name: camera_behind_table_6
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [-100.0, 0.0, 0.0]}
        translation: [-0.75, -0.45, 0.3]

- add_model:
    name: camera8
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_behind_table_6
    child: camera8::base

- add_frame:
    name: camera_behind_table_7
    X_PF:
        base_frame: table
        rotation: !Rpy { deg: [-100.0, 0.0, 0.0]}
        translation: [0.75, -0.45, 0.3]

- add_model:
    name: camera9
    file: package://manipulation/camera_box.sdf

- add_weld:
    parent: camera_behind_table_7
    child: camera9::base

"""
    return scenario_data

def add_mobile_iiwa(scenario_data) -> str:
    scenario_data += """
- add_model:
    name: mobile_iiwa
    file: package://manipulation/mobile_iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.2]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
        iiwa_base_x: [1]
        iiwa_base_y: [1.5]
        iiwa_base_z: [0]
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: mobile_iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90]}
    """
# - add_model:
#     name: wsg
#     file: package://drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
# - add_weld:
#     parent: mobile_iiwa::iiwa_link_7
#     child: wsg::body
#     X_PC:
#         translation: [0, 0, 0.09]
#         rotation: !Rpy { deg: [90, 0, 90]}
#     """
    return scenario_data

def add_cameras(scenario_data):
    return """
cameras:
    camera0:
        name: table_camera_0
        depth: True
        X_PB:
            base_frame: camera_table_above
    
    camera1:
        name: table_camera_1
        depth: True
        X_PB:
            base_frame: camera_table_left
    
    camera2:
        name: table_camera_2
        depth: True
        X_PB:
            base_frame: camera_table_right

    camera3:
        name: table_camera_3
        depth: True
        X_PB:
            base_frame: camera_behind_table_1
    
    camera4:
        name: table_camera_4
        depth: True
        X_PB:
            base_frame: camera_behind_table_2
    
    camera5:
        name: table_camera_5
        depth: True
        X_PB:
            base_frame: camera_behind_table_3
    
    camera6:
        name: table_camera_6
        depth: True
        X_PB:
            base_frame: camera_behind_table_4
    
    camera7:
        name: table_camera_7
        depth: True
        X_PB:
            base_frame: camera_behind_table_5
    camera8:
        name: table_camera_8
        depth: True
        X_PB:
            base_frame: camera_behind_table_6
    camera9:
        name: table_camera_9
        depth: True
        X_PB:
            base_frame: camera_behind_table_7
""" + scenario_data

def add_model_driver(scenario_data):
    return scenario_data + """
model_drivers:
    mobile_iiwa: !InverseDynamicsDriver {}
    wsg: !SchunkWsgDriver {}
"""

def get_library_scenario_data(cameras: bool = True) -> str:
    """Add all objects to library environment"""
    
    scenario_data = get_empty_scenario_data()
    #scenario_data = add_ground_floor(scenario_data)
    scenario_data = add_shelves(scenario_data)
    scenario_data = add_table(scenario_data)
    scenario_data = add_books(scenario_data)
    scenario_data = add_mobile_iiwa(scenario_data)
    scenario_data = add_camera_visual(scenario_data)
    if cameras:
        scenario_data = add_cameras(scenario_data)
    scenario_data = add_model_driver(scenario_data)
    
    return scenario_data

def get_library_scenario_data_without_robot() -> str:
    """Add all objects to library environment"""
    
    scenario_data = get_empty_scenario_data()
    #scenario_data = add_ground_floor(scenario_data)
    scenario_data = add_shelves(scenario_data)
    scenario_data = add_table(scenario_data)
    scenario_data = add_books(scenario_data)
    scenario_data = add_camera_visual(scenario_data)
    scenario_data = add_cameras(scenario_data)

    return scenario_data