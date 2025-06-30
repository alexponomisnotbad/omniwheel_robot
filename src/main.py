import mujoco
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class RobotConfig:
    WHEEL_RADIUS: float = 0.0309  # Wheel radius (m)
    WHEEL_DISTANCE: float = 0.0472  # Distance from robot center to wheel (m)
    BLUE_AREA_THRESHOLD: float = 0.1045
    CAMERA_MATRIX: np.ndarray = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
    DIST_COEFFS: np.ndarray = np.zeros((5,), dtype=np.float32)
    WIDTH: int = 640
    HEIGHT: int = 480
    DT: float = 20 / 1000.0  # Time step in seconds
    CAMERA_ID: str = "top_view"
    BLUE_LOWER: np.ndarray = np.array([110, 150, 150])
    BLUE_UPPER: np.ndarray = np.array([120, 255, 255])

class ColorDetector:
    def __init__(self, lower_bound: np.ndarray, upper_bound: np.ndarray):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def detect(self, hsv_frame: np.ndarray, config: RobotConfig) -> Optional[np.ndarray]:
        mask = cv2.inRange(hsv_frame, self.lower_bound, self.upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) <= 100:
            return None

        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        pixel = np.array([[cx, cy]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pixel, config.CAMERA_MATRIX, config.DIST_COEFFS)
        z, camera_pos = 0, np.array([0, 0, 3])
        return np.array([
            -undistorted[0, 0, 0] * (z - camera_pos[2]) + camera_pos[0],
            -undistorted[0, 0, 1] * (z - camera_pos[2]) + camera_pos[1],
            z
        ])

class CameraProcessor:
    def __init__(self, model: mujoco.MjModel, config: RobotConfig):
        self.renderer = mujoco.Renderer(model, height=config.HEIGHT, width=config.WIDTH)
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, config.CAMERA_ID)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

    def process_frame(self, data: mujoco.MjData) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]:
        self.renderer.update_scene(data, camera=self.camera_id)
        rgb = self.renderer.render()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        robot_pos, robot_angle = None, None
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.08, RobotConfig.CAMERA_MATRIX, RobotConfig.DIST_COEFFS)
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, RobotConfig.CAMERA_MATRIX, RobotConfig.DIST_COEFFS, rvecs[i], tvecs[i], 0.02)
                robot_pos = tvecs[i][0, :]  # x, y, z
                rot_matrix, _ = cv2.Rodrigues(rvecs[i][0, :])
                robot_angle = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])

        return frame, robot_pos, robot_angle

class RobotController:
    def __init__(self, model: mujoco.MjModel, config: RobotConfig):
        self.model = model
        self.config = config
        self.kinematic_matrix = (1 / config.WHEEL_RADIUS) * np.array([
            [-config.WHEEL_DISTANCE, 1, 0],
            [-config.WHEEL_DISTANCE, -0.5, -np.sqrt(3) / 2],
            [-config.WHEEL_DISTANCE, -0.5, np.sqrt(3) / 2]
        ])
        self.motor_indices = {
            'wheel1_speed': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'wheel1_speed'),
            'wheel2_speed': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'wheel2_speed'),
            'wheel3_speed': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'wheel3_speed'),
        }
        self.prev_delta = np.zeros(2)
        self.integral_error = np.zeros(2)

    def compute_control(self, robot_pos: np.ndarray, target_pos: np.ndarray, data: mujoco.MjData) -> Tuple[bool, np.ndarray]:
        delta = target_pos[:2] - robot_pos[:2]
        distance = np.linalg.norm(delta)
        if distance < self.config.BLUE_AREA_THRESHOLD:
            for key in self.motor_indices:
                data.ctrl[self.motor_indices[key]] = 0
            return True, np.zeros(3)

        Kp, Kd, Ki = 1.0, 2.0, 5.0
        max_speed, min_speed = 0.2, 0.01
        max_integral = 0.5

        self.integral_error += delta * self.config.DT
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        derivative = (delta - self.prev_delta) / self.config.DT
        self.prev_delta = delta.copy()

        v = Kp * delta + Kd * derivative + Ki * self.integral_error
        v = np.clip(v, -max_speed, max_speed)
        v = np.where(np.abs(v) < min_speed, min_speed * np.sign(v), v)

        velocities = np.array([0.0, v[1], v[0]])  # omega, v_y, v_x
        wheel_speeds = self.kinematic_matrix @ velocities
        for i, key in enumerate(self.motor_indices):
            data.ctrl[self.motor_indices[key]] = wheel_speeds[i]

        return False, wheel_speeds

class TrajectoryPlotter:
    def __init__(self):
        self.robot_positions: List[np.ndarray] = []
        self.wheel_speeds: List[np.ndarray] = []

    def log(self, robot_pos: Optional[np.ndarray], wheel_speeds: np.ndarray):
        self.robot_positions.append(robot_pos.copy() if robot_pos is not None else np.array([np.nan, np.nan, np.nan]))
        self.wheel_speeds.append(wheel_speeds)

    def plot(self):
        robot_positions = np.array(self.robot_positions)
        robot_positions[:, 0] = gaussian_filter1d(robot_positions[:, 0], sigma=2)
        robot_positions[:, 1] = -gaussian_filter1d(robot_positions[:, 1], sigma=2)

        plt.figure(figsize=(8, 8))
        plt.plot(robot_positions[:, 0], robot_positions[:, 1], label="Robot trajectory", color="blue")
        plt.title("Robot trajectory towards blue area")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.legend()
        plt.grid()
        plt.axis('equal')

        wheel_speeds = np.array(self.wheel_speeds)
        plt.figure(figsize=(12, 6))
        for i, label in enumerate(["Wheel 1", "Wheel 2", "Wheel 3"]):
            plt.plot(wheel_speeds[:, i], label=label)
        plt.title("Wheel speeds")
        plt.xlabel("Iteration")
        plt.ylabel("Speed")
        plt.legend()
        plt.grid()
        plt.show()

def main():
    model = mujoco.MjModel.from_xml_path("robot_soccer_kit_red_and_blue/scene.xml")
    data = mujoco.MjData(model)
    config = RobotConfig()
    color_detector = ColorDetector(config.BLUE_LOWER, config.BLUE_UPPER)
    camera_processor = CameraProcessor(model, config)
    robot_controller = RobotController(model, config)
    plotter = TrajectoryPlotter()

    while True:
        mujoco.mj_step(model, data, nstep=10)
        frame, robot_pos, _ = camera_processor.process_frame(data)
        blue_pos = color_detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), config)

        if blue_pos is not None:
            cv2.circle(frame, (int(blue_pos[0] * 100 + config.WIDTH / 2), int(-blue_pos[1] * 100 + config.HEIGHT / 2)), 5, (255, 0, 0), -1)

        reached, wheel_speeds = robot_controller.compute_control(robot_pos, blue_pos, data) if robot_pos is not None and blue_pos is not None else (False, np.zeros(3))
        plotter.log(robot_pos, wheel_speeds)

        cv2.imshow('rgb_image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or reached:
            cv2.imwrite("camera_output_end.jpg", frame)
            break

    cv2.destroyAllWindows()
    plotter.plot()

if __name__ == "__main__":
    main()