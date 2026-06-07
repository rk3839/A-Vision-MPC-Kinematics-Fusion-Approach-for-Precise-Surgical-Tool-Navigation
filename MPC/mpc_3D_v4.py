"""
Collision avoidance using Nonlinear Model-Predictive Control
3D point-mass robot model with simplified cost function (no F, no theta)

State: [x, y, z]
Control: [vx, vy, vz]
"""

import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import pandas as pd

# Simulation parameters
SIM_TIME = 20.0
TIMESTEP = 0.2
NUMBER_OF_TIMESTEPS = int(SIM_TIME / TIMESTEP)

# Robot parameters
ROBOT_RADIUS = 0.3  # Rb in the paper
SAFETY_MARGIN = 0.4  # beta in the paper
VMAX = 1.0  # Maximum velocity in each axis
VMIN = -1.0  # Minimum velocity

# NMPC parameters
HORIZON_LENGTH = 8  # N prediction steps
NMPC_TIMESTEP = 0.3  # Delta T

# Weight matrices - simplified (no F, using Q for both running and terminal cost)
Q = np.diag([20.0, 20.0, 15.0])  # State error weight for [x, y, z]
R = np.diag([0.01, 0.01, 0.01])  # Control effort weight for [vx, vy, vz]

# Obstacle definitions for 14x14x10 workspace
OBSTACLES = [
    # Static obstacles [x, y, z, vx, vy, vz]
    #{'pos': np.array([5.0, 3.0, 4.0]), 'vel': np.array([0.0, 0.0, 0.0]), 'static': True},
    #{'pos': np.array([8.0, 7.0, 5.0]), 'vel': np.array([0.0, 0.0, 0.0]), 'static': True},
    {'pos': np.array([6.5, 6.0, 2.0]), 'vel': np.array([0.0, 0.0, 0.0]), 'static': True},
    # Dynamic obstacles
    #{'pos': np.array([2.0, 2.0, 2.0]), 'vel': np.array([0.25, 0.25, 0.15]), 'static': False},
    {'pos': np.array([5.25, 8.5, 2.0]), 'vel': np.array([0.0, -0.6, 0.0]), 'static': False},
]


def create_obstacle_trajectories():
    """Generate obstacle positions over time"""
    obstacle_history = np.zeros((NUMBER_OF_TIMESTEPS, len(OBSTACLES), 3))
    
    for t in range(NUMBER_OF_TIMESTEPS):
        for i, obs in enumerate(OBSTACLES):
            if obs['static']:
                obstacle_history[t, i, :] = obs['pos']
            else:
                # Dynamic obstacles with bounds checking
                new_pos = obs['pos'] + obs['vel'] * t * TIMESTEP
                # Keep obstacles within workspace
                new_pos[0] = np.clip(new_pos[0], 1.0, 13.0)  # x
                new_pos[1] = np.clip(new_pos[1], 1.0, 13.0)  # y
                new_pos[2] = np.clip(new_pos[2], 1.0, 9.0)   # z
                obstacle_history[t, i, :] = new_pos
    
    return obstacle_history


def kinematic_model(state, control):
    """
    Simple point-mass kinematic model
    state: [x, y, z]
    control: [vx, vy, vz]
    state_dot = control (direct velocity control)
    """
    return control


def integrate_state(state, control, dt):
    """
    Simple integration for point-mass model
    x(t+dt) = x(t) + v*dt
    """
    new_state = state + control * dt
    return new_state


def predict_trajectory(initial_state, control_sequence, dt):
    """
    Predict robot trajectory over horizon
    control_sequence: [vx1, vy1, vz1, vx2, vy2, vz2, ...]
    Returns: array of shape (HORIZON_LENGTH, 3) containing states
    """
    trajectory = np.zeros((HORIZON_LENGTH, 3))
    state = initial_state.copy()
    
    for i in range(HORIZON_LENGTH):
        control = control_sequence[i*3:(i+1)*3]
        state = integrate_state(state, control, dt)
        trajectory[i, :] = state
    
    return trajectory


def predict_obstacles(obstacles_current, horizon_steps, dt):
    """Predict obstacle positions over prediction horizon"""
    predictions = []
    
    for obs in obstacles_current:
        obs_pred = np.zeros((horizon_steps, 3))
        for i in range(horizon_steps):
            predicted_pos = obs['pos'] + obs['vel'] * (i + 1) * dt
            # Keep predictions within workspace
            predicted_pos[0] = np.clip(predicted_pos[0], 1.0, 13.0)
            predicted_pos[1] = np.clip(predicted_pos[1], 1.0, 13.0)
            predicted_pos[2] = np.clip(predicted_pos[2], 1.0, 9.0)
            obs_pred[i, :] = predicted_pos
        predictions.append(obs_pred)
    
    return predictions


def running_cost(state_error, control_error):
    """
    Running cost l(xe, ue) = xe^T*Q*xe + ue^T*R*ue
    """
    cost = state_error.T @ Q @ state_error + control_error.T @ R @ control_error
    return cost


def terminal_cost(state_error):
    """
    Terminal cost P(xe) = 0.5 * xe^T * Q * xe
    Using Q instead of separate F matrix (simplified)
    """
    cost = 0.5 * state_error.T @ Q @ state_error
    return cost


def collision_constraint_value(robot_pos, obstacle_pos):
    """
    Collision avoidance constraint: distance >= (Rb + beta)
    Returns: distance - (Rb + beta)
    """
    distance = np.linalg.norm(robot_pos - obstacle_pos)
    min_distance = ROBOT_RADIUS + SAFETY_MARGIN
    return distance - min_distance


def barrier_function(distance, min_dist, barrier_gain=10.0):
    """
    Barrier function for collision avoidance
    Returns high cost when getting close to obstacles
    """
    if distance <= min_dist:
        return 1e6  # Very high penalty for collision
    elif distance < min_dist + 1.0:
        # Smooth barrier that increases as we approach min_dist
        return barrier_gain / (distance - min_dist)
    else:
        return 0.0


def cost_function(control_sequence, current_state, reference_state, obstacle_predictions):
    """
    Total cost function J = Sum of running costs + terminal cost + barriers
    """
    # Predict robot trajectory
    trajectory = predict_trajectory(current_state, control_sequence, NMPC_TIMESTEP)
    
    # Initialize cost
    total_cost = 0.0
    
    # Reference control (zero for set-point stabilization)
    reference_control = np.zeros(3)
    
    min_distance = ROBOT_RADIUS + SAFETY_MARGIN
    
    # Running cost over prediction horizon
    for i in range(HORIZON_LENGTH):
        state_error = reference_state - trajectory[i, :]
        control = control_sequence[i*3:(i+1)*3]
        control_error = reference_control - control
        
        # Add running cost
        total_cost += running_cost(state_error, control_error) * NMPC_TIMESTEP
        
        # Add barrier function for each obstacle
        for obs_pred in obstacle_predictions:
            distance = np.linalg.norm(trajectory[i, :] - obs_pred[i, :])
            
            # Strong barrier function
            barrier_cost = barrier_function(distance, min_distance, barrier_gain=50.0)
            total_cost += barrier_cost
            
            # Additional quadratic penalty for violations
            if distance < min_distance:
                violation = min_distance - distance
                total_cost += 1000.0 * (violation ** 2)
        
        # Workspace boundaries soft constraint
        if trajectory[i, 0] < 1.0 or trajectory[i, 0] > 13.0:
            total_cost += 100.0 * ((min(abs(trajectory[i, 0] - 1.0), abs(13.0 - trajectory[i, 0]))) ** 2)
        if trajectory[i, 1] < 1.0 or trajectory[i, 1] > 13.0:
            total_cost += 100.0 * ((min(abs(trajectory[i, 1] - 1.0), abs(13.0 - trajectory[i, 1]))) ** 2)
        if trajectory[i, 2] < 1.0 or trajectory[i, 2] > 9.0:
            total_cost += 100.0 * ((min(abs(trajectory[i, 2] - 1.0), abs(9.0 - trajectory[i, 2]))) ** 2)
    
    # Terminal cost (using Q, not F)
    terminal_state_error = reference_state - trajectory[-1, :]
    total_cost += terminal_cost(terminal_state_error)
    
    return total_cost


def compute_control(current_state, reference_state, obstacles_current):
    """
    Solve NMPC optimization problem
    """
    # Predict obstacle positions
    obstacle_predictions = predict_obstacles(obstacles_current, HORIZON_LENGTH, NMPC_TIMESTEP)
    
    # Initial guess - small velocities toward goal
    direction_to_goal = reference_state - current_state
    if np.linalg.norm(direction_to_goal) > 0.1:
        direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal) * 0.2
    else:
        direction_to_goal = np.zeros(3)
    
    u0 = np.tile(direction_to_goal, HORIZON_LENGTH)
    
    # Bounds on control inputs
    lower_bounds = np.array([VMIN, VMIN, VMIN] * HORIZON_LENGTH)
    upper_bounds = np.array([VMAX, VMAX, VMAX] * HORIZON_LENGTH)
    bounds = Bounds(lower_bounds, upper_bounds)
    
    # Define cost function
    def cost(u):
        return cost_function(u, current_state, reference_state, obstacle_predictions)
    
    # Solve optimization problem
    result = minimize(
        cost,
        u0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 150, 'ftol': 1e-6, 'disp': False}
    )
    
    if result.success or result.fun < 1e10:
        optimal_control = result.x[:3]  # Take first control action
    else:
        print(f"Warning: Optimization struggled")
        optimal_control = np.zeros(3)
    
    return optimal_control


def simulate(animate=True):
    """Main simulation loop"""
    # Initial and goal states for 14x14x10 workspace
    initial_state = np.array([2.0, 6.0, 2.0])  # [x, y, z]
    goal_state = np.array([8.5, 6.0, 2.0])    # Target position
    
    # Generate obstacle trajectories
    obstacle_trajectories = create_obstacle_trajectories()
    
    # Initialize storage
    state_history = np.zeros((NUMBER_OF_TIMESTEPS, 3))
    control_history = np.zeros((NUMBER_OF_TIMESTEPS, 3))
    
    current_state = initial_state.copy()
    
    print("Starting NMPC simulation...")
    print(f"Robot model: 3D point-mass (x, y, z)")
    print(f"Control: Linear velocities (vx, vy, vz)")
    print(f"Initial state: {initial_state}")
    print(f"Goal state: {goal_state}")
    print(f"Workspace: [0, 14] x [0, 14] x [0, 10]")
    print(f"Cost function: Using Q for both running and terminal cost (no F)")
    
    # Setup 2D animation (top view)
    if animate:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left plot: Top view (XY plane)
        ax1.set_xlim(0, 14)
        ax1.set_ylim(0, 14)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X (m)', fontsize=12)
        ax1.set_ylabel('Y (m)', fontsize=12)
        ax1.set_title('Top View (XY Plane)', fontsize=13, fontweight='bold')
        
        # Plot goal
        ax1.plot(goal_state[0], goal_state[1], 'r*', markersize=25, label='Goal', zorder=5)
        
        # Robot patch
        robot_patch = Circle((initial_state[0], initial_state[1]), ROBOT_RADIUS, 
                            facecolor='green', edgecolor='black', linewidth=2, zorder=4, label='Robot')
        ax1.add_patch(robot_patch)
        
        # Trajectory line
        trajectory_line, = ax1.plot([], [], 'b-', linewidth=2.5, label='Path', alpha=0.8)
        
        # Obstacle patches
        obstacle_patches = []
        colors = ['dimgray', 'dimgray', 'dimgray', 'cyan', 'orange']
        for i, obs in enumerate(OBSTACLES):
            obs_patch = Circle((obs['pos'][0], obs['pos'][1]), 
                             ROBOT_RADIUS + SAFETY_MARGIN,
                             facecolor=colors[i], edgecolor='black', 
                             alpha=0.7, linewidth=2, zorder=3)
            ax1.add_patch(obs_patch)
            obstacle_patches.append(obs_patch)
            
            label = 'Static' if obs['static'] else 'Dynamic'
            ax1.text(obs['pos'][0], obs['pos'][1] - 0.8, f'{label} {i+1}', 
                   ha='center', fontsize=9, fontweight='bold')
        
        ax1.legend(loc='upper left', fontsize=10)
        
        # Right plot: Side view (XZ plane)
        ax2.set_xlim(0, 14)
        ax2.set_ylim(0, 5)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('X (m)', fontsize=12)
        ax2.set_ylabel('Z (m)', fontsize=12)
        ax2.set_title('Side View (XZ Plane)', fontsize=13, fontweight='bold')
        
        # Goal in side view
        ax2.plot(goal_state[0], goal_state[2], 'r*', markersize=25, label='Goal', zorder=5)
        
        # Robot in side view
        robot_patch_side = Circle((initial_state[0], initial_state[2]), ROBOT_RADIUS, 
                                 facecolor='green', edgecolor='black', linewidth=2, zorder=4)
        ax2.add_patch(robot_patch_side)
        
        # Trajectory in side view
        trajectory_line_side, = ax2.plot([], [], 'b-', linewidth=2.5, alpha=0.8)
        
        # Obstacles in side view
        obstacle_patches_side = []
        for i, obs in enumerate(OBSTACLES):
            obs_patch_side = Circle((obs['pos'][0], obs['pos'][2]), 
                                   ROBOT_RADIUS + SAFETY_MARGIN,
                                   facecolor=colors[i], edgecolor='black', 
                                   alpha=0.7, linewidth=2, zorder=3)
            ax2.add_patch(obs_patch_side)
            obstacle_patches_side.append(obs_patch_side)
        
        ax2.legend(loc='upper left', fontsize=10)
        
        # Status text
        status_text = fig.text(0.5, 0.02, '', ha='center', fontsize=11,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.draw()
        plt.pause(0.1)
    
    for t in range(NUMBER_OF_TIMESTEPS):
        # Current obstacles
        obstacles_current = []
        for i, obs in enumerate(OBSTACLES):
            obstacles_current.append({
                'pos': obstacle_trajectories[t, i, :],
                'vel': obs['vel'],
                'static': obs['static']
            })
        
        # Compute optimal control
        optimal_control = compute_control(current_state, goal_state, obstacles_current)
        
        # Apply control and update state
        current_state = integrate_state(current_state, optimal_control, TIMESTEP)
        
        # Keep robot within workspace
        current_state[0] = np.clip(current_state[0], 0.5, 13.5)
        current_state[1] = np.clip(current_state[1], 0.5, 13.5)
        current_state[2] = np.clip(current_state[2], 0.5, 9.5)
        
        # Store history
        state_history[t, :] = current_state
        control_history[t, :] = optimal_control
        
        # Check minimum distance to obstacles
        min_obs_distance = float('inf')
        for obs in obstacles_current:
            dist = np.linalg.norm(current_state - obs['pos'])
            min_obs_distance = min(min_obs_distance, dist)
        
        # Update animation
        if animate:
            # Update top view
            robot_patch.center = (current_state[0], current_state[1])
            trajectory_line.set_data(state_history[:t+1, 0], state_history[:t+1, 1])
            
            for i, obs_patch in enumerate(obstacle_patches):
                obs_patch.center = (obstacle_trajectories[t, i, 0], obstacle_trajectories[t, i, 1])
            
            # Update side view
            robot_patch_side.center = (current_state[0], current_state[2])
            trajectory_line_side.set_data(state_history[:t+1, 0], state_history[:t+1, 2])
            
            for i, obs_patch_side in enumerate(obstacle_patches_side):
                obs_patch_side.center = (obstacle_trajectories[t, i, 0], obstacle_trajectories[t, i, 2])
            
            # Update status
            error = np.linalg.norm(goal_state - current_state)
            collision_status = "SAFE" if min_obs_distance > ROBOT_RADIUS + SAFETY_MARGIN else "TOO CLOSE!"
            status_text.set_text(f'Step: {t+1}/{NUMBER_OF_TIMESTEPS} | Time: {t*TIMESTEP:.1f}s | '
                               f'Pos: ({current_state[0]:.1f}, {current_state[1]:.1f}, {current_state[2]:.1f}) | '
                               f'Error: {error:.2f}m | Min dist: {min_obs_distance:.2f}m | {collision_status}')
            
            plt.draw()
            plt.pause(0.01)
        
        # Print progress
        if t % 10 == 0:
            error = np.linalg.norm(goal_state - current_state)
            print(f"Step {t}/{NUMBER_OF_TIMESTEPS}, Error: {error:.3f}m, Min dist: {min_obs_distance:.3f}m")
    
    # Final statistics
    final_error = goal_state - current_state
    print(f"\nSimulation Complete!")
    print(f"Final error: x={final_error[0]:.4f}m, y={final_error[1]:.4f}m, z={final_error[2]:.4f}m")
    
    min_distance_overall = float('inf')
    for t in range(NUMBER_OF_TIMESTEPS):
        for i in range(len(OBSTACLES)):
            dist = np.linalg.norm(state_history[t, :] - obstacle_trajectories[t, i, :])
            min_distance_overall = min(min_distance_overall, dist)
    
    required_clearance = ROBOT_RADIUS + SAFETY_MARGIN
    print(f"Minimum clearance: {min_distance_overall:.3f}m (required: {required_clearance:.3f}m)")
    if min_distance_overall >= required_clearance:
        print("✓ SUCCESS: No collisions!")
    else:
        print("✗ WARNING: Too close to obstacles!")
    
    if animate:
        plt.ioff()
        input("\nPress Enter to close and show final plots...")
        plt.close()
    
    return state_history, obstacle_trajectories


def save_to_excel(state_history, obstacle_trajectories, filename='robot_paths.xlsx'):
    """
    Save robot path and dynamic obstacles paths to Excel file
    """
    # Create time array
    time = np.arange(NUMBER_OF_TIMESTEPS) * TIMESTEP
    
    # Create a Pandas Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: Robot Path
        robot_df = pd.DataFrame({
            'Time (s)': time,
            'Robot_X (m)': state_history[:, 0],
            'Robot_Y (m)': state_history[:, 1],
            'Robot_Z (m)': state_history[:, 2]
        })
        robot_df.to_excel(writer, sheet_name='Robot_Path', index=False)
        
        # Sheet 2: All Obstacles (combined)
        obstacles_data = {'Time (s)': time}
        
        # Add each obstacle's path
        for i, obs in enumerate(OBSTACLES):
            obs_label = f"Obstacle_{i+1}"
            obs_type = "Static" if obs['static'] else "Dynamic"
            
            obstacles_data[f'{obs_label}_Type'] = [obs_type] * NUMBER_OF_TIMESTEPS
            obstacles_data[f'{obs_label}_X (m)'] = obstacle_trajectories[:, i, 0]
            obstacles_data[f'{obs_label}_Y (m)'] = obstacle_trajectories[:, i, 1]
            obstacles_data[f'{obs_label}_Z (m)'] = obstacle_trajectories[:, i, 2]
        
        all_obstacles_df = pd.DataFrame(obstacles_data)
        all_obstacles_df.to_excel(writer, sheet_name='All_Obstacles', index=False)
        
        # Sheet 3: Dynamic Obstacles Only
        dynamic_data = {'Time (s)': time}
        dynamic_count = 0
        
        for i, obs in enumerate(OBSTACLES):
            if not obs['static']:  # Only dynamic obstacles
                dynamic_count += 1
                dynamic_label = f"Dynamic_{dynamic_count}"
                dynamic_data[f'{dynamic_label}_X (m)'] = obstacle_trajectories[:, i, 0]
                dynamic_data[f'{dynamic_label}_Y (m)'] = obstacle_trajectories[:, i, 1]
                dynamic_data[f'{dynamic_label}_Z (m)'] = obstacle_trajectories[:, i, 2]
                dynamic_data[f'{dynamic_label}_Vx (m/s)'] = [obs['vel'][0]] * NUMBER_OF_TIMESTEPS
                dynamic_data[f'{dynamic_label}_Vy (m/s)'] = [obs['vel'][1]] * NUMBER_OF_TIMESTEPS
                dynamic_data[f'{dynamic_label}_Vz (m/s)'] = [obs['vel'][2]] * NUMBER_OF_TIMESTEPS
        
        if dynamic_count > 0:
            dynamic_df = pd.DataFrame(dynamic_data)
            dynamic_df.to_excel(writer, sheet_name='Dynamic_Obstacles_Only', index=False)
        
        # Sheet 4: Summary Statistics
        goal = np.array([8.5, 6.0, 2.0])
        final_error = goal - state_history[-1, :]
        total_distance_traveled = np.sum(np.linalg.norm(np.diff(state_history, axis=0), axis=1))
        
        # Calculate minimum distance to each obstacle
        min_distances = []
        for i in range(len(OBSTACLES)):
            min_dist = float('inf')
            for t in range(NUMBER_OF_TIMESTEPS):
                dist = np.linalg.norm(state_history[t, :] - obstacle_trajectories[t, i, :])
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)
        
        summary_data = {
            'Parameter': [
                'Initial X (m)', 'Initial Y (m)', 'Initial Z (m)',
                'Goal X (m)', 'Goal Y (m)', 'Goal Z (m)',
                'Final X (m)', 'Final Y (m)', 'Final Z (m)',
                'Final Error X (m)', 'Final Error Y (m)', 'Final Error Z (m)',
                'Total Distance Traveled (m)',
                'Simulation Time (s)',
                'Number of Obstacles',
                'Number of Dynamic Obstacles',
                'Robot Radius (m)',
                'Safety Margin (m)',
                'Required Clearance (m)'
            ],
            'Value': [
                state_history[0, 0], state_history[0, 1], state_history[0, 2],
                goal[0], goal[1], goal[2],
                state_history[-1, 0], state_history[-1, 1], state_history[-1, 2],
                final_error[0], final_error[1], final_error[2],
                total_distance_traveled,
                SIM_TIME,
                len(OBSTACLES),
                sum(1 for obs in OBSTACLES if not obs['static']),
                ROBOT_RADIUS,
                SAFETY_MARGIN,
                ROBOT_RADIUS + SAFETY_MARGIN
            ]
        }
        
        # Add minimum distances to each obstacle
        for i, min_dist in enumerate(min_distances):
            summary_data['Parameter'].append(f'Min Distance to Obstacle {i+1} (m)')
            summary_data['Value'].append(min_dist)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\n✓ Data saved to '{filename}'")
    print(f"  - Sheet 1: Robot_Path (robot trajectory)")
    print(f"  - Sheet 2: All_Obstacles (all obstacles)")
    print(f"  - Sheet 3: Dynamic_Obstacles_Only (dynamic obstacles)")
    print(f"  - Sheet 4: Summary (statistics)")

    """Final visualization"""
    fig = plt.figure(figsize=(18, 6))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2], 
            'b-', linewidth=3, label='Robot path')
    ax1.scatter(state_history[0, 0], state_history[0, 1], state_history[0, 2], 
               c='green', s=100, marker='o', label='Start')
    ax1.scatter(8.5, 6.0, 2.0, c='red', s=200, marker='*', label='Goal')
    
    # Plot obstacles
    colors = ['dimgray', 'dimgray', 'dimgray', 'cyan', 'orange']
    for i, obs in enumerate(OBSTACLES):
        if obs['static']:
            ax1.scatter(obs['pos'][0], obs['pos'][1], obs['pos'][2], 
                       c=colors[i], s=200, alpha=0.6, marker='o')
        else:
            ax1.plot(obstacle_trajectories[:, i, 0], obstacle_trajectories[:, i, 1], 
                    obstacle_trajectories[:, i, 2], '--', color=colors[i], linewidth=2, alpha=0.6)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory', fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 14)
    ax1.set_zlim(0, 5)
    
    # Position errors
    ax2 = fig.add_subplot(132)
    time = np.arange(NUMBER_OF_TIMESTEPS) * TIMESTEP
    goal = np.array([8.5, 6.0, 2.0])
    
    ax2.plot(time, np.abs(goal[0] - state_history[:, 0]), linewidth=2, label='|x_error|')
    ax2.plot(time, np.abs(goal[1] - state_history[:, 1]), linewidth=2, label='|y_error|')
    ax2.plot(time, np.abs(goal[2] - state_history[:, 2]), linewidth=2, label='|z_error|')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Absolute Error (m)')
    ax2.set_title('Tracking Errors', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Total error
    ax3 = fig.add_subplot(133)
    total_error = np.linalg.norm(goal - state_history, axis=1)
    ax3.plot(time, total_error, linewidth=2, color='purple')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Total Error (m)')
    ax3.set_title('Euclidean Distance to Goal', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nmpc_results.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run simulation with animation
    state_history, obstacle_trajectories = simulate(animate=True)
    
    # Save data to Excel
    save_to_excel(state_history, obstacle_trajectories, filename='robot_paths.xlsx')
    
    # Show final plots
    plot_results(state_history, obstacle_trajectories)
    print("\nVisualization saved to 'nmpc_results.png'")
    print("All data saved to 'robot_paths.xlsx'")