"""
Live Camera-based NMPC Path Planning
Detects green object as robot position and plans collision-free path
Press SPACEBAR to trigger path planning
"""

import numpy as np
from scipy.optimize import minimize, Bounds
import cv2
import time
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings

# Suppress scipy optimization warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.optimize')

# ============================================================================
# USER CONFIGURABLE PARAMETERS
# ============================================================================

# Image plane dimensions
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Z-axis range
Z_MIN = 0.0
Z_MAX = 5.0

# Goal/Target location (can be changed by user) - [x, y, z]
GOAL_POSITION = np.array([550.0, 100.0, 4.0])  # [x, y, z] in image coordinates + z

# Static obstacles (can be changed by user) - [x, y, z, radius]
STATIC_OBSTACLES = [
    {'pos': np.array([200.0, 150.0, 2.0]), 'radius': 40.0, 'color': (128, 128, 128)},
    {'pos': np.array([400.0, 300.0, 3.0]), 'radius': 50.0, 'color': (100, 100, 100)},
    {'pos': np.array([150.0, 350.0, 1.5]), 'radius': 35.0, 'color': (120, 120, 120)},
]

# Dynamic obstacles (can be changed by user) - [x, y, z, radius, vx, vy, vz]
DYNAMIC_OBSTACLES = [
    {'pos': np.array([100.0, 100.0, 1.0]), 'radius': 30.0, 
     'vel': np.array([20.0, 20.0, 0.3]), 'color': (255, 200, 0)},
    {'pos': np.array([500.0, 400.0, 4.0]), 'radius': 35.0, 
     'vel': np.array([-15.0, -15.0, -0.2]), 'color': (255, 150, 50)},
]

# Default robot position if green object not detected - [x, y, z]
DEFAULT_ROBOT_POSITION = np.array([50.0, 400.0, 0.5])

# Green color detection range in HSV (for robot)
GREEN_LOWER = np.array([40, 40, 40])
GREEN_UPPER = np.array([80, 255, 255])

# Blue color detection range in HSV (for dynamic obstacle)
BLUE_LOWER = np.array([100, 100, 50])
BLUE_UPPER = np.array([130, 255, 255])

# Blue object settings
BLUE_OBJECT_RADIUS = 30.0
BLUE_OBJECT_Z = 2.5  # Fixed z-coordinate for blue detected obstacle

# ============================================================================
# NMPC PARAMETERS
# ============================================================================

ROBOT_RADIUS = 20.0
SAFETY_MARGIN = 25.0
VMAX = 100.0  # Increased for faster movement
VMIN = -100.0
VZ_MAX = 30.0  # Z velocity limit
VZ_MIN = -30.0

HORIZON_LENGTH = 8  # Increased for better lookahead    # N prediction steps
NMPC_TIMESTEP = 0.15    # Delta T
MAX_PLANNING_TIMESTEPS = 200  # Increased to allow reaching distant goals

Q = np.diag([20.0, 20.0, 15.0])  # State error weight for [x, y, z]
R = np.diag([0.005, 0.005, 0.01])  # Control effort weight for [vx, vy, vz]

# ============================================================================
# COLOR OBJECT DETECTION
# ============================================================================

def detect_green_object(frame):
    """Detect green object in frame and return its center position with default z"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
            # Get center of contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Add z coordinate (use default robot z)
                return np.array([float(cx), float(cy), DEFAULT_ROBOT_POSITION[2]]), True
    
    return None, False

def detect_blue_object(frame):
    """Detect blue object in frame and return its center position for dynamic obstacle"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
            # Get center of contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Add z coordinate
                return np.array([float(cx), float(cy), BLUE_OBJECT_Z]), True
    
    return None, False

# ============================================================================
# OBSTACLE MANAGEMENT
# ============================================================================

def update_dynamic_obstacles(obstacles, dt):
    """Update positions of dynamic obstacles in 3D"""
    updated = []
    for obs in obstacles:
        new_pos = obs['pos'] + obs['vel'] * dt
        
        # Bounce off boundaries in X and Y
        if new_pos[0] < obs['radius'] or new_pos[0] > IMAGE_WIDTH - obs['radius']:
            obs['vel'][0] = -obs['vel'][0]
            new_pos[0] = np.clip(new_pos[0], obs['radius'], IMAGE_WIDTH - obs['radius'])
        
        if new_pos[1] < obs['radius'] or new_pos[1] > IMAGE_HEIGHT - obs['radius']:
            obs['vel'][1] = -obs['vel'][1]
            new_pos[1] = np.clip(new_pos[1], obs['radius'], IMAGE_HEIGHT - obs['radius'])
        
        # Bounce off boundaries in Z
        if new_pos[2] < Z_MIN or new_pos[2] > Z_MAX:
            obs['vel'][2] = -obs['vel'][2]
            new_pos[2] = np.clip(new_pos[2], Z_MIN, Z_MAX)
        
        updated.append({
            'pos': new_pos,
            'radius': obs['radius'],
            'vel': obs['vel'],
            'color': obs['color']
        })
    
    return updated

def get_all_obstacles(static_obs, dynamic_obs, blue_detected_obs=None):
    """Combine static, dynamic, and blue detected obstacles for path planning"""
    all_obs = []
    
    for obs in static_obs:
        all_obs.append({
            'pos': obs['pos'],
            'radius': obs['radius'],
            'static': True
        })
    
    for obs in dynamic_obs:
        all_obs.append({
            'pos': obs['pos'],
            'radius': obs['radius'],
            'static': False,
            'vel': obs['vel']
        })
    
    # Add blue detected obstacle as static
    if blue_detected_obs is not None:
        all_obs.append({
            'pos': blue_detected_obs['pos'],
            'radius': blue_detected_obs['radius'],
            'static': True  # Treat as static during planning
        })
    
    return all_obs

# ============================================================================
# NMPC PATH PLANNING
# ============================================================================

def predict_trajectory(initial_state, control_sequence, dt):
    """Predict robot trajectory in 3D"""
    trajectory = np.zeros((HORIZON_LENGTH, 3))
    state = initial_state.copy()
    
    for i in range(HORIZON_LENGTH):
        control = control_sequence[i*3:(i+1)*3]
        state = state + control * dt
        trajectory[i, :] = state
    
    return trajectory

def predict_obstacles(obstacles, horizon_steps, dt):
    """Predict obstacle positions in 3D"""
    predictions = []
    
    for obs in obstacles:
        obs_pred = np.zeros((horizon_steps, 3))
        for i in range(horizon_steps):
            if obs['static']:
                obs_pred[i, :] = obs['pos']
            else:
                predicted_pos = obs['pos'] + obs['vel'] * (i + 1) * dt
                predicted_pos[0] = np.clip(predicted_pos[0], 0, IMAGE_WIDTH)
                predicted_pos[1] = np.clip(predicted_pos[1], 0, IMAGE_HEIGHT)
                predicted_pos[2] = np.clip(predicted_pos[2], Z_MIN, Z_MAX)
                obs_pred[i, :] = predicted_pos
        predictions.append((obs_pred, obs['radius']))
    
    return predictions

def barrier_function(distance, min_dist):
    """Barrier function for collision avoidance"""
    if distance <= min_dist:
        return 1e6
    elif distance < min_dist + 50:
        return 100.0 / (distance - min_dist)
    else:
        return 0.0

def cost_function(control_sequence, current_state, goal_state, obstacle_predictions):
    """NMPC cost function for 3D"""
    trajectory = predict_trajectory(current_state, control_sequence, NMPC_TIMESTEP)
    
    total_cost = 0.0
    reference_control = np.zeros(3)
    
    for i in range(HORIZON_LENGTH):
        state_error = goal_state - trajectory[i, :]
        control = control_sequence[i*3:(i+1)*3]
        control_error = reference_control - control
        
        # Running cost
        total_cost += (state_error.T @ Q @ state_error + 
                      control_error.T @ R @ control_error) * NMPC_TIMESTEP
        
        # Obstacle avoidance
        for obs_pred, obs_radius in obstacle_predictions:
            distance = np.linalg.norm(trajectory[i, :] - obs_pred[i, :])
            min_distance = ROBOT_RADIUS + obs_radius + SAFETY_MARGIN
            
            barrier_cost = barrier_function(distance, min_distance)
            total_cost += barrier_cost
            
            if distance < min_distance:
                violation = min_distance - distance
                total_cost += 2000.0 * (violation ** 2)
        
        # Boundary constraints
        if trajectory[i, 0] < 30 or trajectory[i, 0] > IMAGE_WIDTH - 30:
            total_cost += 500.0
        if trajectory[i, 1] < 30 or trajectory[i, 1] > IMAGE_HEIGHT - 30:
            total_cost += 500.0
        if trajectory[i, 2] < Z_MIN or trajectory[i, 2] > Z_MAX:
            total_cost += 500.0
    
    # Terminal cost
    terminal_error = goal_state - trajectory[-1, :]
    total_cost += 0.5 * terminal_error.T @ Q @ terminal_error
    
    return total_cost

def compute_control(current_state, goal_state, obstacles):
    """Compute optimal control for 3D"""
    obstacle_predictions = predict_obstacles(obstacles, HORIZON_LENGTH, NMPC_TIMESTEP)
    
    direction_to_goal = goal_state - current_state
    if np.linalg.norm(direction_to_goal) > 0.1:
        direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal) * 20.0
    else:
        direction_to_goal = np.zeros(3)
    
    u0 = np.tile(direction_to_goal, HORIZON_LENGTH)
    
    lower_bounds = np.array([VMIN, VMIN, VZ_MIN] * HORIZON_LENGTH)
    upper_bounds = np.array([VMAX, VMAX, VZ_MAX] * HORIZON_LENGTH)
    bounds = Bounds(lower_bounds, upper_bounds)
    
    def cost(u):
        return cost_function(u, current_state, goal_state, obstacle_predictions)
    
    result = minimize(cost, u0, method='SLSQP', bounds=bounds,
                     options={'maxiter': 100, 'ftol': 1e-6, 'disp': False})
    
    if result.success or result.fun < 1e10:
        return result.x[:3]
    else:
        return np.zeros(3)

def plan_path_threaded(start_pos, goal_pos, obstacles, result_dict):
    """
    Plan complete path from start to goal in 3D (runs in separate thread)
    result_dict: dictionary to store results and progress
    """
    result_dict['is_planning'] = True
    result_dict['progress'] = 0
    result_dict['current_path'] = [start_pos.copy()]
    
    path = [start_pos.copy()]
    current_state = start_pos.copy()
    
    planning_start = time.time()
    
    # Calculate distance to goal
    initial_distance = np.linalg.norm(goal_pos - start_pos)
    
    # Adaptive number of steps based on distance
    max_steps = max(MAX_PLANNING_TIMESTEPS, int(initial_distance / 5))
    
    for step in range(max_steps):
        # Check if reached goal (with tolerance)
        distance_to_goal = np.linalg.norm(current_state - goal_pos)
        if distance_to_goal < 15:  # Goal reached threshold
            path.append(goal_pos.copy())
            break
        
        # Update dynamic obstacles for prediction
        current_obstacles = []
        for obs in obstacles:
            if obs['static']:
                current_obstacles.append(obs)
            else:
                # Predict where dynamic obstacle will be
                future_pos = obs['pos'] + obs['vel'] * step * NMPC_TIMESTEP
                future_pos[0] = np.clip(future_pos[0], 0, IMAGE_WIDTH)
                future_pos[1] = np.clip(future_pos[1], 0, IMAGE_HEIGHT)
                future_pos[2] = np.clip(future_pos[2], Z_MIN, Z_MAX)
                current_obstacles.append({
                    'pos': future_pos,
                    'radius': obs['radius'],
                    'static': False,
                    'vel': obs['vel']
                })
        
        # Compute control
        control = compute_control(current_state, goal_pos, current_obstacles)
        
        # Update state
        current_state = current_state + control * NMPC_TIMESTEP
        current_state[0] = np.clip(current_state[0], 0, IMAGE_WIDTH)
        current_state[1] = np.clip(current_state[1], 0, IMAGE_HEIGHT)
        current_state[2] = np.clip(current_state[2], Z_MIN, Z_MAX)
        
        path.append(current_state.copy())
        
        # Update progress for live visualization
        result_dict['progress'] = (step / max_steps) * 100
        result_dict['current_path'] = path.copy()
    
    planning_time = time.time() - planning_start
    
    # Calculate path length
    path_length = 0.0
    for i in range(1, len(path)):
        path_length += np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
    
    # Calculate final error
    final_error = np.linalg.norm(np.array(path[-1]) - goal_pos)
    
    # Store final results
    result_dict['is_planning'] = False
    result_dict['final_path'] = path
    result_dict['planning_time'] = planning_time
    result_dict['path_length'] = path_length
    result_dict['final_error'] = final_error
    result_dict['completed'] = True

# ============================================================================
# 3D VISUALIZATION
# ============================================================================

def create_3d_plot(path, static_obs, dynamic_obs, goal_pos, robot_pos, blue_detected_obs=None):
    """Create 3D visualization of the path"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set labels and limits
    ax.set_xlabel('X (pixels)', fontsize=10)
    ax.set_ylabel('Y (pixels)', fontsize=10)
    ax.set_zlabel('Z (height)', fontsize=10)
    ax.set_xlim(0, IMAGE_WIDTH)
    ax.set_ylim(0, IMAGE_HEIGHT)
    ax.set_zlim(Z_MIN, Z_MAX)
    ax.set_title('3D Path Planning', fontsize=12, fontweight='bold')
    
    # Plot path
    if path is not None and len(path) > 1:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
               'purple', linewidth=3, alpha=0.8, label='Path')
        
        # Mark path points
        for i in range(0, len(path), max(1, len(path)//10)):
            ax.scatter(path[i][0], path[i][1], path[i][2],
                      c='purple', s=20, alpha=0.6)
    
    # Plot robot
    ax.scatter(robot_pos[0], robot_pos[1], robot_pos[2],
              c='green', s=200, marker='o', label='Robot', edgecolors='black', linewidths=2)
    
    # Plot goal
    ax.scatter(goal_pos[0], goal_pos[1], goal_pos[2],
              c='red', s=300, marker='*', label='Goal', edgecolors='black', linewidths=2)
    
    # Plot static obstacles
    for i, obs in enumerate(static_obs):
        ax.scatter(obs['pos'][0], obs['pos'][1], obs['pos'][2],
                  c='gray', s=obs['radius']*10, marker='o', alpha=0.6,
                  label='Static Obs' if i == 0 else '', edgecolors='black', linewidths=1.5)
    
    # Plot dynamic obstacles
    for i, obs in enumerate(dynamic_obs):
        ax.scatter(obs['pos'][0], obs['pos'][1], obs['pos'][2],
                  c='orange', s=obs['radius']*10, marker='o', alpha=0.7,
                  label='Dynamic Obs' if i == 0 else '', edgecolors='black', linewidths=1.5)
    
    # Plot blue detected obstacle
    if blue_detected_obs is not None:
        ax.scatter(blue_detected_obs['pos'][0], blue_detected_obs['pos'][1], blue_detected_obs['pos'][2],
                  c='blue', s=blue_detected_obs['radius']*10, marker='o', alpha=0.8,
                  label='Blue Detected', edgecolors='black', linewidths=2)
    
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=20, azim=45)
    
    # Convert plot to image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    plot_image = np.asarray(buf)
    plt.close(fig)
    
    # Convert RGBA to BGR for OpenCV
    plot_image = cv2.cvtColor(plot_image, cv2.COLOR_RGBA2BGR)
    
    # Resize to fit nicely
    plot_image = cv2.resize(plot_image, (640, 480))
    
    return plot_image

def draw_obstacles(frame, static_obs, dynamic_obs, blue_detected_obs=None):
    """Draw obstacles on frame"""
    # Draw static obstacles
    for obs in static_obs:
        cv2.circle(frame, (int(obs['pos'][0]), int(obs['pos'][1])),
                  int(obs['radius']), obs['color'], -1)
        cv2.circle(frame, (int(obs['pos'][0]), int(obs['pos'][1])),
                  int(obs['radius']), (0, 0, 0), 2)
        cv2.putText(frame, 'S', (int(obs['pos'][0])-8, int(obs['pos'][1])+8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw predefined dynamic obstacles
    for obs in dynamic_obs:
        cv2.circle(frame, (int(obs['pos'][0]), int(obs['pos'][1])),
                  int(obs['radius']), obs['color'], -1)
        cv2.circle(frame, (int(obs['pos'][0]), int(obs['pos'][1])),
                  int(obs['radius']), (0, 0, 0), 2)
        cv2.putText(frame, 'D', (int(obs['pos'][0])-8, int(obs['pos'][1])+8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw blue detected obstacle
    if blue_detected_obs is not None:
        cv2.circle(frame, (int(blue_detected_obs['pos'][0]), int(blue_detected_obs['pos'][1])),
                  int(blue_detected_obs['radius']), (255, 100, 0), -1)
        cv2.circle(frame, (int(blue_detected_obs['pos'][0]), int(blue_detected_obs['pos'][1])),
                  int(blue_detected_obs['radius']), (0, 0, 0), 2)
        cv2.putText(frame, 'BLUE', (int(blue_detected_obs['pos'][0])-20, int(blue_detected_obs['pos'][1])+8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"z={blue_detected_obs['pos'][2]:.1f}", 
                   (int(blue_detected_obs['pos'][0])-20, int(blue_detected_obs['pos'][1])-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def draw_goal(frame, goal_pos):
    """Draw goal position (XY projection)"""
    cv2.drawMarker(frame, (int(goal_pos[0]), int(goal_pos[1])),
                  (0, 0, 255), cv2.MARKER_STAR, 30, 3)
    cv2.circle(frame, (int(goal_pos[0]), int(goal_pos[1])), 15, (0, 0, 255), 2)
    cv2.putText(frame, f'GOAL (z={goal_pos[2]:.1f})', 
               (int(goal_pos[0])-45, int(goal_pos[1])-25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def draw_robot(frame, robot_pos):
    """Draw robot position (XY projection)"""
    cv2.circle(frame, (int(robot_pos[0]), int(robot_pos[1])),
              int(ROBOT_RADIUS), (0, 255, 0), -1)
    cv2.circle(frame, (int(robot_pos[0]), int(robot_pos[1])),
              int(ROBOT_RADIUS), (0, 0, 0), 2)
    # Show z coordinate
    cv2.putText(frame, f'z={robot_pos[2]:.1f}',
               (int(robot_pos[0])-20, int(robot_pos[1])+30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def draw_path(frame, path):
    """Draw planned path (XY projection)"""
    if len(path) > 1:
        points = np.array([[int(p[0]), int(p[1])] for p in path], np.int32)
        cv2.polylines(frame, [points], False, (255, 0, 255), 2)
        
        # Draw points along path
        for i, point in enumerate(path):
            if i % 5 == 0:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (255, 0, 255), -1)

def draw_info(frame, robot_detected, blue_detected, planning_time, path_length, final_error, 
              is_planning, planning_progress):
    """Draw information overlay"""
    info_y = 30
    
    # Title
    cv2.putText(frame, 'NMPC Path Planning', (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    info_y += 30
    
    # Detection status for green (robot)
    status = "Green (Robot): DETECTED" if robot_detected else "Green (Robot): NOT DETECTED"
    color = (0, 255, 0) if robot_detected else (0, 0, 255)
    cv2.putText(frame, status, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    info_y += 25
    
    # Detection status for blue (obstacle)
    status_blue = "Blue (Obstacle): DETECTED" if blue_detected else "Blue (Obstacle): NOT DETECTED"
    color_blue = (255, 100, 0) if blue_detected else (0, 0, 255)
    cv2.putText(frame, status_blue, (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_blue, 2)
    info_y += 25
    
    # Planning status
    if is_planning:
        cv2.putText(frame, f'Planning... {planning_progress:.0f}%', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        info_y += 25
    elif planning_time > 0:
        cv2.putText(frame, f'Planning Time: {planning_time*1000:.1f} ms', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(frame, f'Path Length: {path_length:.1f} px', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(frame, f'Reach Error: {final_error:.1f} px', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        info_y += 25
    else:
        cv2.putText(frame, 'Press SPACEBAR to plan path', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        info_y += 25
    
    cv2.putText(frame, 'Press Q to quit', (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    """Main application loop"""
    print("="*70)
    print("LIVE 3D NMPC PATH PLANNING WITH CAMERA")
    print("="*70)
    print(f"Image dimensions: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Z-axis range: [{Z_MIN}, {Z_MAX}]")
    print(f"Goal position: ({GOAL_POSITION[0]:.0f}, {GOAL_POSITION[1]:.0f}, {GOAL_POSITION[2]:.1f})")
    print(f"Static obstacles: {len(STATIC_OBSTACLES)}")
    print(f"Dynamic obstacles: {len(DYNAMIC_OBSTACLES)}")
    print("\nControls:")
    print("  - SPACEBAR: Plan/Replan path")
    print("  - Q: Quit")
    print("="*70 + "\n")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize variables
    planned_path = None
    robot_position = DEFAULT_ROBOT_POSITION.copy()
    robot_detected = False
    blue_detected_obstacle = None
    blue_detected = False
    planning_time = 0
    path_length = 0
    final_error = 0
    
    # Threading variables
    planning_thread = None
    planning_result = {
        'is_planning': False,
        'progress': 0,
        'current_path': [],
        'final_path': None,
        'planning_time': 0,
        'path_length': 0,
        'final_error': 0,
        'completed': False
    }
    
    # Dynamic obstacles state
    dynamic_obs = [obs.copy() for obs in DYNAMIC_OBSTACLES]
    last_update_time = time.time()
    
    print("Camera started. Press SPACEBAR to plan path...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Update dynamic obstacles
        current_time = time.time()
        dt = current_time - last_update_time
        if dt > 0.05:  # Update at 20 Hz
            dynamic_obs = update_dynamic_obstacles(dynamic_obs, dt)
            last_update_time = current_time
        
        # Detect green object (robot)
        detected_pos, detected = detect_green_object(frame)
        if detected:
            robot_position = detected_pos
            robot_detected = True
        else:
            robot_position = DEFAULT_ROBOT_POSITION
            robot_detected = False
        
        # Detect blue object (dynamic obstacle)
        blue_pos, blue_det = detect_blue_object(frame)
        if blue_det:
            blue_detected_obstacle = {
                'pos': blue_pos,
                'radius': BLUE_OBJECT_RADIUS,
                'color': (255, 100, 0)
            }
            blue_detected = True
        else:
            blue_detected_obstacle = None
            blue_detected = False
        
        # Check if planning completed
        if planning_result['completed']:
            planned_path = planning_result['final_path']
            planning_time = planning_result['planning_time']
            path_length = planning_result['path_length']
            final_error = planning_result['final_error']
            planning_result['completed'] = False
            
            print(f"Planning completed!")
            print(f"  - Planning time: {planning_time*1000:.1f} ms")
            print(f"  - Path length: {path_length:.1f} pixels")
            print(f"  - Reach error: {final_error:.1f} pixels")
            print(f"{'='*50}\n")
        
        # Draw 2D camera view
        draw_obstacles(frame, STATIC_OBSTACLES, dynamic_obs, blue_detected_obstacle)
        draw_goal(frame, GOAL_POSITION)
        draw_robot(frame, robot_position)
        
        # Draw path (either final or in-progress)
        if planning_result['is_planning'] and len(planning_result['current_path']) > 0:
            draw_path(frame, planning_result['current_path'])
        elif planned_path is not None:
            draw_path(frame, planned_path)
        
        # Draw info
        draw_info(frame, robot_detected, blue_detected, planning_time, path_length, 
                 final_error, planning_result['is_planning'], planning_result['progress'])
        
        # Create 3D visualization
        current_path = planning_result['current_path'] if planning_result['is_planning'] else planned_path
        plot_3d = create_3d_plot(current_path, STATIC_OBSTACLES, dynamic_obs, 
                                 GOAL_POSITION, robot_position, blue_detected_obstacle)
        
        # Combine camera view and 3D plot side by side
        combined = np.hstack([frame, plot_3d])
        
        # Display combined frame
        cv2.imshow('NMPC Path Planning - Camera + 3D View', combined)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break
        
        elif key == ord(' '):  # Spacebar
            # Don't start new planning if already planning
            if planning_result['is_planning']:
                print("Planning already in progress...")
                continue
            
            print(f"\n{'='*50}")
            print(f"Planning 3D path from ({robot_position[0]:.1f}, {robot_position[1]:.1f}, {robot_position[2]:.1f}) to goal...")
            
            # Reset result dict
            planning_result = {
                'is_planning': True,
                'progress': 0,
                'current_path': [robot_position.copy()],
                'final_path': None,
                'planning_time': 0,
                'path_length': 0,
                'final_error': 0,
                'completed': False
            }
            
            # Get all obstacles for planning (snapshot)
            all_obstacles_snapshot = [obs.copy() for obs in dynamic_obs]
            all_obstacles = get_all_obstacles(STATIC_OBSTACLES, all_obstacles_snapshot, blue_detected_obstacle)
            
            # Start planning in separate thread
            planning_thread = threading.Thread(
                target=plan_path_threaded,
                args=(robot_position.copy(), GOAL_POSITION, all_obstacles, planning_result)
            )
            planning_thread.daemon = True
            planning_thread.start()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed.")

if __name__ == "__main__":
    main()