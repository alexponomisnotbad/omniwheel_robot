import mujoco
import mujoco.viewer
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from collections import deque

model = mujoco.MjModel.from_xml_path("robot_soccer_kit_red_and_blue/scene.xml")
data = mujoco.MjData(model)


width, height = 640, 480
renderer = mujoco.Renderer(model, height=height, width=width)


motor_indices = {
    'wheel1_speed': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,'wheel1_speed'),
    'wheel2_speed': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,'wheel2_speed'),
    'wheel3_speed': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,'wheel3_speed'),
    }

# Параметры робота
R = 0.0309  # Радиус колеса (м)
L = 0.0472   # Расстояние от центра робота до колеса (м)
kinematic_matrix = (1/R) * np.array([
    [-L, 1, 0],
    [-L, -0.5, -np.sqrt(3)/2],
    [-L, -0.5, np.sqrt(3)/2]
])

# Параметры для ArUco-метки
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Цветовые диапазоны для синей и красной области в HSV
blue_lower = np.array([110, 150, 150]) 
blue_upper = np.array([120, 255, 255])
red_lower = np.array([0, 50, 50])      # Нижняя граница для красного (0-10 градусов)
red_upper = np.array([10, 255, 255])   # Верхняя граница для красного
red_lower2 = np.array([170, 50, 50])   # Вторая граница для красного (170-180 градусов)
red_upper2 = np.array([180, 255, 255]) # Вторая граница для красного


blue_area_threshold = 0.1
speed_scale = 0.1
dt = 20
max_iteration = 1000

# Параметры камеры (нужны для вычисления позиции ArUco-метки)
# Замените на реальные параметры вашей камеры (матрица камеры и коэффициенты дисторсии)
camera_matrix = np.array([[width, 0, width/2], [0, width, height/2], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5,), dtype=np.float32)  # Предполагаем отсутствие дисторсии

# очереди для координат
blue_pos_buffer = deque(maxlen=5)
red_pos_buffer = deque(maxlen=5)


output_dir = "output"
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view")     
i = 0
while True:
    i+=1
    
    mujoco.mj_step(model, data, nstep=10)


    # Рендеринг изображения
    renderer.update_scene(data, camera=camera_id)
    rgb = renderer.render()

    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if i == 1:
        sucsess = cv2.imwrite("camera_output_1.jpg", frame)
        if sucsess:
            print(f"Сохранение первого изображение")
        else:
            print("Сохранение первого изображения не удачно ")


    

    corners, ids, rejected = detector.detectMarkers(gray)

    robot_pos = None
    robot_angle = None
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # Оценка позы метки через ArucoDetector
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.02)
            robot_pos = tvecs[i][0, :]  # Позиция робота (x, y, z)
            rvec = rvecs[i][0, :]
            rot_matrix, _ = cv2.Rodrigues(rvec)
            robot_angle = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Обнаружение красной области ---

    red_mask1 = cv2.inRange(hsv, red_lower, red_upper)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2
    
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hsv, red_contours, -1, (0, 255, 255), 3)

    red_area_3d = None
    if red_contours:
        largest_contour = max(red_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                pixel = np.array([[cx, cy]], dtype=np.float32)
                # print(f"координаты в пикселях красной области {pixel}")
                unistorted_red = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)
                z = 0
                camera_pos = np.array([0, 0, 3])
                red_area_3d = np.array([
                    unistorted_red[0,0,0] * (z - camera_pos[2]) + camera_pos[0],
                    unistorted_red[0,0,1] * (z - camera_pos[2]) + camera_pos[1],
                    z
                ])
                red_pos_buffer.append(red_area_3d)
                cv2.circle(frame,(int(cx), int(cy)), 5, (255,255,0), -1)


    # --- Обнаружение синей области ---

    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hsv, blue_contours, -1, (0, 255, 255), 3)
    


    blue_area_3d = None
    if blue_contours:
        largest_contour_blue = max(blue_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour_blue) > 100:
            M = cv2.moments(largest_contour_blue)
            if M["m00"] != 0:
                cx_blue = int(M["m10"] / M["m00"])
                cy_blue = int(M["m01"] / M["m00"])
                pixel_blue = np.array([[cx_blue, cy_blue]], dtype=np.float32)
                # print(f"координаты в пикселях синей области {pixel_blue}")
                unistorted_blue = cv2.undistortPoints(pixel_blue, camera_matrix, dist_coeffs)
                z = 0
                camera_pos = np.array([0, 0, 3])
                blue_area_3d = np.array([
                    unistorted_blue[0,0,0] * (z - camera_pos[2]) + camera_pos[0],
                    unistorted_blue[0,0,1] * (z - camera_pos[2]) + camera_pos[1],
                    z
                ])
                blue_pos_buffer.append(blue_area_3d)
                cv2.circle(frame, (int(cx_blue), int(cy_blue)), 5, (255, 0, 0), -1)
    
    cv2.imshow('rgb_image', frame)
    key = cv2.waitKey(1) & 0xFF

    


    if robot_pos is not None:
        print(f"Robot position: {robot_pos}, angle: {robot_angle:.2f} rad")
    if blue_area_3d is not None:
        print(f"Blue area position (pixels): {blue_area_3d}")
    # if red_area_3d is not None:
    #     print(f"Red area position (pixels): {red_area_3d}")
    

# Управление роботом
    if robot_pos is not None and blue_area_3d is not None:
        delta = blue_area_3d[:2] - robot_pos[:2]  # Только x, y
        distance_to_blue = np.linalg.norm(delta)
        if distance_to_blue < blue_area_threshold:

            cv2.imwrite(os.path.join(output_dir, "camera_output_end.png"), frame)
            break

        v_x = delta[0] / 10
        v_y = delta[1] / 10
        omega = 0

        velocities = np.array([omega, v_y, v_x])
        wheel_speeds = kinematic_matrix @ velocities
        data.ctrl[motor_indices['wheel1_speed']] = wheel_speeds[0]
        data.ctrl[motor_indices['wheel2_speed']] = wheel_speeds[1]
        data.ctrl[motor_indices['wheel3_speed']] = wheel_speeds[2]
    else:
        data.ctrl[motor_indices['wheel1_speed']] = 0
        data.ctrl[motor_indices['wheel2_speed']] = 0
        data.ctrl[motor_indices['wheel3_speed']] = 0



    
    if i > max_iteration:
        print("Достигнут лимит итераций")
        sucsess = cv2.imwrite("camera_output_end.png", frame)
        if sucsess:
            print(f"Сохранение последнего изображение")
        else:
            print("Сохранение последнего изображения не удачно ")
        break
cv2.destroyAllWindows()