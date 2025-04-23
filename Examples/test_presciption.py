#!/usr/bin/env python3

import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import cv2
import imageio
import random
import threading
import queue
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

# For creating OpenCV windows on some systems:
os.environ["QT_QPA_PLATFORM"] = "xcb"

# ------------------------
# ManipulaPy imports
# (Ensure these modules are installed/in your PYTHONPATH)
# ------------------------
from ManipulaPy.sim import Simulation
from ManipulaPy.ManipulaPy_data.xarm import urdf_file as ur5_urdf_file
from ManipulaPy.vision import Vision
from ManipulaPy.perception import Perception

# ------------------------
# Global concurrency flags/queues
# ------------------------
stop_perception = False
visualization_queue = queue.Queue()
detected_obstacles = []  # Store obstacle centers (in world coordinates)

# =============================================================================
# Helper: World-to-Camera Extrinsic
# =============================================================================
def compute_world_to_camera_extrinsic(camera_config):
    """
    Builds a 4x4 transformation from world -> camera, given translation + Euler angles in degrees.
    """
    R_cam_to_world = R.from_euler('xyz', camera_config["rotation"], degrees=True).as_matrix()
    t_cam_to_world = np.array(camera_config["translation"]).reshape((3, 1))
    T_cam_to_world = np.eye(4)
    T_cam_to_world[:3, :3] = R_cam_to_world
    T_cam_to_world[:3, 3] = t_cam_to_world.flatten()

    # Invert for world->camera
    return np.linalg.inv(T_cam_to_world)

# =============================================================================
# PyBullet Setup Helpers
# =============================================================================
def init_pybullet():
    """Ensures only one PyBullet GUI connection is created."""
    try:
        try:
            p.disconnect()
        except Exception:
            pass
        # Connect using DIRECT first then switch to GUI
        _ = p.connect(p.DIRECT)
        p.disconnect()
        physics_client = p.connect(p.GUI)
        return physics_client
    except p.error as e:
        print(f"Failed to connect to PyBullet GUI: {e}")
        return None

def create_random_obstacles(num_obstacles=5):
    """Create random small cubes (using cube_small.urdf) as obstacles in mid-air."""
    obstacles = []
    positions = []
    for _ in range(num_obstacles):
        x = random.uniform(0.3, 0.7)
        y = random.uniform(-0.3, 0.3)
        z = random.uniform(0.1, 0.5)
        obs_id = p.loadURDF("cube_small.urdf", [x, y, z], useFixedBase=True)
        obstacles.append(obs_id)
        positions.append([x, y, z])
    return obstacles, positions

def create_camera_visual(config, color):
    """Create a visual marker for a camera, plus a line for its forward direction."""
    size = 0.05
    orientation = p.getQuaternionFromEuler([np.radians(r) for r in config["rotation"]])
    vis_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size/2, size, size/2],
        rgbaColor=color
    )
    col_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size/2, size, size/2]
    )
    camera_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=config["translation"],
        baseOrientation=orientation
    )
    # Compute a forward point (for visualization, using the x-axis as forward)
    rot = R.from_euler('xyz', [np.radians(r) for r in config["rotation"]])
    direction = rot.apply([1, 0, 0])
    end_point = np.array(config["translation"]) + direction * 0.2
    p.addUserDebugLine(
        lineFromXYZ=config["translation"],
        lineToXYZ=end_point,
        lineColorRGB=color[:3],
        lineWidth=2
    )
    return camera_id

# =============================================================================
# Toy Collision & Potential Field
# (For demonstration: a toy collision check in joint space)
# =============================================================================
def is_in_collision(joint_config, obstacles):
    """
    Toy collision check: interpret 'joint_config' and each obstacle as points in the same space,
    and if distance < threshold then flag a collision.
    """
    collision_threshold = 0.2
    for obs in obstacles:
        if len(obs) == len(joint_config):
            dist = np.linalg.norm(joint_config - obs)
            if dist < collision_threshold:
                return True
    return False

def compute_potential_field_gradient(current_config, goal_config, obstacles):
    """
    Toy potential field in joint space:
      - Attracts current configuration toward the goal.
      - Repels away from obstacles if within a threshold.
    """
    rep_gain = 1.0
    att_gain = 1.0
    threshold = 0.2

    # Attractive component:
    grad_attractive = (current_config - goal_config) * att_gain

    # Repulsive component:
    grad_repulsive = np.zeros_like(current_config)
    for obs in obstacles:
        if len(obs) != len(current_config):
            continue
        diff = current_config - obs
        dist = np.linalg.norm(diff)
        if dist < threshold:
            rep_strength = rep_gain * (1.0/dist - 1.0/threshold)
            grad_repulsive += rep_strength * diff / (dist**3 + 1e-8)
    return grad_attractive + grad_repulsive

def plan_trajectory_with_avoidance(start_joints, end_joints, obstacle_centers, num_points=100):
    """
    Collision-aware trajectory planning:
      1) Naively linearly interpolate between start and goal.
      2) Check for collisions and adjust waypoints using a potential field.
    """
    start = np.array(start_joints, dtype=np.float32)
    end   = np.array(end_joints,  dtype=np.float32)
    trajectory = []

    # Linear interpolation:
    for i in range(num_points + 1):
        alpha = i / float(num_points)
        q = (1 - alpha)*start + alpha*end
        trajectory.append(q.copy())

    # Adjust waypoints that are in collision:
    max_iter = 100
    learning_rate = 0.01
    for idx, q in enumerate(trajectory):
        if is_in_collision(q, obstacle_centers):
            for _ in range(max_iter):
                grad = compute_potential_field_gradient(q, end, obstacle_centers)
                q_new = q - learning_rate * grad
                for j in range(len(q_new)):
                    q_new[j] = np.clip(q_new[j], -2*np.pi, 2*np.pi)
                if not is_in_collision(q_new, obstacle_centers):
                    q = q_new
                    break
                q = q_new
            trajectory[idx] = q.copy()
    return trajectory

# =============================================================================
# Extended Vision Class: No Class Filtering
# =============================================================================
class PyBulletVision(Vision):
    """
    Extended Vision class that uses YOLO to detect objects but does not filter by class.
    All bounding boxes become 'obstacles.'
    """
    def __init__(self, camera_configs, stereo_configs, use_pybullet_debug=False, show_plot=False, physics_client=None):
        super().__init__(camera_configs, stereo_configs, use_pybullet_debug, show_plot)
        self.physics_client = physics_client
        try:
            self.yolo_model = YOLO("yolov8m.pt")
            self.logger.info("YOLO model initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {e}")
            self.yolo_model = None

    def capture_image(self, camera_index=0):
        """
        Capture an RGB image and a depth map from PyBullet.
        Instead of a hardcoded target, we now use a target that roughly centers the obstacles.
        """
        if camera_index not in range(len(self.cameras)):
            self.logger.error(f"Camera index {camera_index} not found.")
            return None, None

        cfg = self.cameras[camera_index]
        width, height = 640, 480
        # Instead of a fixed [0,0,0.5], set a target that centers the obstacles:
        target = np.array([0.6, 0, 0.3])
        up_vector = [0, 0, 1]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cfg["translation"],
            cameraTargetPosition=target,
            cameraUpVector=up_vector
        )
        aspect = width / float(height)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=cfg["fov"],
            aspect=aspect,
            nearVal=cfg["near"],
            farVal=cfg["far"]
        )
        _, _, rgba, depth_buf, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
        rgb = rgba[..., :3]
        depth = np.array(depth_buf, dtype=np.float32).reshape((height, width))
        near, far = cfg["near"], cfg["far"]
        depth = near + (far - near) * depth
        return rgb, depth

# =============================================================================
# Modified Simulation Class
# =============================================================================
class ModifiedSimulation(Simulation):
    """
    Modified Simulation class that uses an existing PyBullet client.
    """
    def __init__(self, urdf_file_path, joint_limits, physics_client,
                 torque_limits=None, time_step=0.01, real_time_factor=1.0):
        self.urdf_file_path = ur5_urdf_file
        self.joint_limits = joint_limits
        self.torque_limits = torque_limits
        self.time_step = time_step
        self.real_time_factor = real_time_factor
        self.logger = self.setup_logger()
        self.physics_client = physics_client
        self.joint_params = []
        self.reset_button = None
        self.home_position = None
        self.setup_simulation_with_client()

    def setup_simulation_with_client(self):
        if self.physics_client is None:
            raise RuntimeError("No PyBullet client provided.")
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_file_path, useFixedBase=True)
        self.non_fixed_joints = [
            i for i in range(p.getNumJoints(self.robot_id))
            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
        ]
        self.home_position = np.zeros(len(self.non_fixed_joints))

# =============================================================================
# Perception + Visualization Threads
# =============================================================================
def perception_processing_thread(vision, perception):
    """
    Capture images, detect and cluster obstacles,
    transform detected centers to world coordinates, and push results for visualization.
    """
    global stop_perception, visualization_queue, detected_obstacles
    while not stop_perception:
        left_rgb, left_depth = vision.capture_image(camera_index=0)
        right_rgb, right_depth = vision.capture_image(camera_index=1)
        if left_rgb is None or right_rgb is None:
            time.sleep(0.1)
            continue

        # Increase eps parameter to allow for more clustering tolerance
        obstacle_points, labels = perception.detect_and_cluster_obstacles(
            camera_index=0,
            depth_threshold=5,
            step=100,
            eps=0.01,
            min_samples=10
        )

        detected_obstacles.clear()
        if obstacle_points.size > 0 and labels.size > 0:
            unique_labels = set(labels) - {-1}
            for label in unique_labels:
                mask = (labels == label)
                cluster_pts = obstacle_points[mask]
                if cluster_pts.size > 0:
                    center_cam = np.mean(cluster_pts, axis=0)
                    # Transform from camera to world coordinates
                    E = vision.cameras[0]["extrinsic_matrix"]
                    T_cam_to_world = np.linalg.inv(E)
                    center_cam_homog = np.append(center_cam, 1)
                    center_world = (T_cam_to_world @ center_cam_homog)[:3]
                    detected_obstacles.append(center_world)
                    dist = np.linalg.norm(center_world)
                    orient = np.degrees(np.arctan2(center_world[1], center_world[0]))
                    print(f"Obstacle {label}: Distance={dist:.2f}m, Orientation={orient:.2f}°")

        visualization_queue.put({
            'left_rgb': left_rgb,
            'right_rgb': right_rgb,
            'obstacle_points': obstacle_points,
            'labels': labels,
            'camera_info': vision.cameras[0]
        })
        time.sleep(0.1)

def visualization_thread():
    """
    Display camera feeds with overlayed bounding circles for detected obstacles.
    """
    global stop_perception, visualization_queue
    cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Right Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Obstacle Detection", cv2.WINDOW_NORMAL)

    while not stop_perception:
        try:
            data = visualization_queue.get(timeout=0.1)
        except queue.Empty:
            time.sleep(0.01)
            continue

        left_disp = cv2.cvtColor(data['left_rgb'], cv2.COLOR_RGB2BGR)
        right_disp = cv2.cvtColor(data['right_rgb'], cv2.COLOR_RGB2BGR)
        obstacle_disp = left_disp.copy()

        if data['obstacle_points'].size > 0 and data['labels'].size > 0:
            unique_labels = set(data['labels'])
            for label in unique_labels:
                if label == -1:
                    continue
                mask = (data['labels'] == label)
                cluster_pts = data['obstacle_points'][mask]
                center = np.mean(cluster_pts, axis=0)
                K = data['camera_info']["intrinsic_matrix"]
                if center[2] > 0:
                    px = K @ (center[:3] / center[2])
                    x2d, y2d = int(px[0]), int(px[1])
                    color = ((label * 50) % 255, (label * 80) % 255, (label * 110) % 255)
                    cv2.circle(obstacle_disp, (x2d, y2d), 10, color, -1)
                    cv2.putText(obstacle_disp, f"Obj {label}",
                                (x2d + 15, y2d),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Left Camera", left_disp)
        cv2.imshow("Right Camera", right_disp)
        cv2.imshow("Obstacle Detection", obstacle_disp)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

# =============================================================================
# Main
# =============================================================================
def main():
    global stop_perception, detected_obstacles
    try:
        p.disconnect()
    except:
        pass
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.81)

    # Set up simulation with the robot
    sim = ModifiedSimulation(
        urdf_file_path=ur5_urdf_file,
        joint_limits=[(-2*np.pi, 2*np.pi)] * 6,
        physics_client=physics_client,
        torque_limits=[(-100, 100)] * 6,
        time_step=0.01,
        real_time_factor=1.0
    )
    sim.initialize_robot()

    # Create random obstacles (using cube_small.urdf)
    obs_ids, obs_positions = create_random_obstacles(num_obstacles=5)
    print("Obstacle positions:", obs_positions)

    # Camera configs
    baseline = 0.2
    cam_distance = 1.2
    cam_height = 1.0
    intrinsic_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0,   0,   1]
    ], dtype=np.float32)

    cam_left_config = {
        "name": "cam_left",
        "translation": [-baseline/2, -cam_distance, cam_height],
        "rotation": [0, -30, 0],
        "fov": 60,
        "near": 0.1,
        "far": 10.0,
        "intrinsic_matrix": intrinsic_matrix,
        "distortion_coeffs": np.zeros(5, dtype=np.float32)
    }
    cam_right_config = {
        "name": "cam_right",
        "translation": [baseline/2, -cam_distance, cam_height],
        "rotation": [0, -30, 0],
        "fov": 60,
        "near": 0.1,
        "far": 10.0,
        "intrinsic_matrix": intrinsic_matrix,
        "distortion_coeffs": np.zeros(5, dtype=np.float32)
    }

    cam_left_config["extrinsic_matrix"]  = compute_world_to_camera_extrinsic(cam_left_config)
    cam_right_config["extrinsic_matrix"] = compute_world_to_camera_extrinsic(cam_right_config)

    # Visual markers for cameras
    create_camera_visual(cam_left_config, [1, 0, 0, 1])   # Red
    create_camera_visual(cam_right_config, [0, 0, 1, 1])   # Blue

    # Create extended Vision + Perception instances
    vision = PyBulletVision(
        camera_configs=[cam_left_config, cam_right_config],
        stereo_configs=(cam_left_config.copy(), cam_right_config.copy()),
        use_pybullet_debug=False,
        show_plot=False,
        physics_client=physics_client
    )
    perception = Perception(vision_instance=vision)

    # Start perception and visualization threads
    stop_perception = False
    t_percep = threading.Thread(target=perception_processing_thread, args=(vision, perception))
    t_percep.daemon = True
    t_percep.start()

    t_vis = threading.Thread(target=visualization_thread)
    t_vis.daemon = True
    t_vis.start()

    print("Perception + Visualization threads running. Waiting 5 seconds for detection...")
    time.sleep(5)

    # Stop threads
    stop_perception = True
    t_percep.join()
    t_vis.join()
    cv2.destroyAllWindows()

    print("Detected obstacle centers (world coordinates):", detected_obstacles)

    # For demonstration, use detected obstacles as obstacles in a toy joint-space planning.
    start_joints = np.zeros(6)
    end_joints   = np.array([np.pi/2, -np.pi/4, np.pi/4, -np.pi/2, np.pi/2, -np.pi/4])
    trajectory = plan_trajectory_with_avoidance(start_joints, end_joints, detected_obstacles, num_points=100)

    frames_cam_left, frames_cam_right = [], []

    try:
        for joints in trajectory:
            sim.set_joint_positions(joints)
            p.stepSimulation()

            rgb_left, _ = vision.capture_image(camera_index=0)
            if rgb_left is not None:
                rgba_left = cv2.cvtColor(rgb_left, cv2.COLOR_RGB2RGBA)
                frames_cam_left.append(rgba_left)
                cv2.imshow("Left Camera Motion", cv2.cvtColor(rgba_left, cv2.COLOR_RGBA2BGR))

            rgb_right, _ = vision.capture_image(camera_index=1)
            if rgb_right is not None:
                rgba_right = cv2.cvtColor(rgb_right, cv2.COLOR_RGB2RGBA)
                frames_cam_right.append(rgba_right)
                cv2.imshow("Right Camera Motion", cv2.cvtColor(rgba_right, cv2.COLOR_RGBA2BGR))

            cv2.waitKey(1)
            time.sleep(sim.time_step / sim.real_time_factor)

        num_steps = 100
        joint_limits = sim.joint_limits
        joint_trajectory = np.linspace(
            np.zeros(len(joint_limits)),
            np.array([-np.pi/2, -np.pi/2, np.pi/4, -np.pi/2, np.pi/2, -np.pi/4]),
            num=num_steps
        )

        print("\nTrajectory complete. Press Ctrl+C or close the window to exit...")
        while True:
            sim.simulate_robot_motion(joint_trajectory)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("User interrupted motion.")
    finally:
        if frames_cam_left:
            imageio.mimsave('left_cam_motion.gif', frames_cam_left, fps=30)
        if frames_cam_right:
            imageio.mimsave('right_cam_motion.gif', frames_cam_right, fps=30)

        sim.close_simulation()
        p.disconnect()  # Disconnect from PyBullet
        cv2.destroyAllWindows()
        print("Script ended, GIFs saved.")

if __name__ == "__main__":
    main()
