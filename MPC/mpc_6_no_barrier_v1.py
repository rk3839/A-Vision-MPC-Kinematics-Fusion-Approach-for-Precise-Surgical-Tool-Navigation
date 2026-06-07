"""
Collision avoidance using Nonlinear Model-Predictive Control
2D point-mass robot model (x, y only - no z, no theta)

State: [x, y]
Control: [vx, vy]

Modified version: Quadratic penalty method (no barrier functions)
"""

import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd
import time

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
Q = np.diag([20.0, 20.0])  # State error weight for [x, y]
R = np.diag([0.01, 0.01])  # Control effort weight for [vx, vy]

# Obstacle definitions for 14x14 workspace
OBSTACLES = [
    # Static obstacles [x, y, vx, vy]
    {'pos': np.array([5.0, 3.0]), 'vel': np.array([0.0, 0.0]), 'static': True},
    {'pos': np.array([7.0, 7.0]), 'vel': np.array([0.0, 0.0]), 'static': True},
    {'pos': np.array([3.0, 9.0]), 'vel': np.array([0.0, 0.0]), 'static': True},
    # Dynamic obstacles
    {'pos': np.array([2.0, 2.0]), 'vel': np.array([0.25, 0.25]), 'static': False},
    {'pos': np.array([12.0, 11.0]), 'vel': np.array([-0.2, -0.2]), 'static': False},
]


def create_obstacle_trajectories():
    """Generate obstacle positions over time"""
    obstacle_history = np.zeros((NUMBER_OF_TIMESTEPS, len(OBSTACLES), 2))
    
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
                obstacle_history[t, i, :] = new_pos
    
    return obstacle_history


def kinematic_model(state, control):
    """
    Simple 2D point-mass kinematic model
    state: [x, y]
    control: [vx, vy]
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
    control_sequence: [vx1, vy1, vx2, vy2, ...]
    Returns: array of shape (HORIZON_LENGTH, 2) containing states
    """
    trajectory = np.zeros((HORIZON_LENGTH, 2))
    state = initial_state.copy()
    
    for i in range(HORIZON_LENGTH):
        control = control_sequence[i*2:(i+1)*2]
        state = integrate_state(state, control, dt)
        trajectory[i, :] = state
    
    return trajectory


def predict_obstacles(obstacles_current, horizon_steps, dt):
    """Predict obstacle positions over prediction horizon"""
    predictions = []
    
    for obs in obstacles_current:
        obs_pred = np.zeros((horizon_steps, 2))
        for i in range(horizon_steps):
            predicted_pos = obs['pos'] + obs['vel'] * (i + 1) * dt
            # Keep predictions within workspace
            predicted_pos[0] = np.clip(predicted_pos[0], 1.0, 13.0)
            predicted_pos[1] = np.clip(predicted_pos[1], 1.0, 13.0)
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


def cost_function(control_sequence, current_state, reference_state, obstacle_predictions):
    """
    Total cost function J = Sum of running costs + terminal cost + collision penalties
    Using quadratic penalty method (not barrier functions)
    """
    # Predict robot trajectory
    trajectory = predict_trajectory(current_state, control_sequence, NMPC_TIMESTEP)
    
    # Initialize cost
    total_cost = 0.0
    
    # Reference control (zero for set-point stabilization)
    reference_control = np.zeros(2)
    
    min_distance = ROBOT_RADIUS + SAFETY_MARGIN
    
    # Running cost over prediction horizon
    for i in range(HORIZON_LENGTH):
        state_error = reference_state - trajectory[i, :]
        control = control_sequence[i*2:(i+1)*2]
        control_error = reference_control - control
        
        # Add running cost
        total_cost += running_cost(state_error, control_error) * NMPC_TIMESTEP
        
        # Collision avoidance using quadratic penalty (NOT barrier function)
        for obs_pred in obstacle_predictions:
            distance = np.linalg.norm(trajectory[i, :] - obs_pred[i, :])
            
            # Quadratic penalty when distance < min_distance
            if distance < min_distance:
                violation = min_distance - distance
                # Very strong quadratic penalty for constraint violation
                total_cost += 10000.0 * (violation ** 2)
            # Additional smooth penalty in safety zone
            elif distance < min_distance + 1.0:
                # Moderate penalty to encourage staying away
                safety_violation = (min_distance + 1.0) - distance
                total_cost += 1000.0 * (safety_violation ** 2)
        
        # Workspace boundaries soft constraint
        if trajectory[i, 0] < 1.0 or trajectory[i, 0] > 13.0:
            total_cost += 100.0 * ((min(abs(trajectory[i, 0] - 1.0), abs(13.0 - trajectory[i, 0]))) ** 2)
        if trajectory[i, 1] < 1.0 or trajectory[i, 1] > 13.0:
            total_cost += 100.0 * ((min(abs(trajectory[i, 1] - 1.0), abs(13.0 - trajectory[i, 1]))) ** 2)
    
    # Terminal cost (using Q, not F)
    terminal_state_error = reference_state - trajectory[-1, :]
    total_cost += terminal_cost(terminal_state_error)
    
    return total_cost


def mpc_controller(current_state, reference_state, obstacles_current, previous_control=None):
    """
    MPC controller that solves the optimization problem
    """
    # Initial guess for control sequence
    if previous_control is not None:
        # Warm-start: shift previous solution
        control_init = np.zeros(HORIZON_LENGTH * 2)
        control_init[:-2] = previous_control[2:]  # Shift forward
        control_init[-2:] = previous_control[-2:]  # Repeat last control
    else:
        control_init = np.zeros(HORIZON_LENGTH * 2)
    
    # Predict obstacle trajectories
    obstacle_predictions = predict_obstacles(obstacles_current, HORIZON_LENGTH, NMPC_TIMESTEP)
    
    # Bounds on control inputs
    bounds = Bounds(
        lb=np.ones(HORIZON_LENGTH * 2) * VMIN,
        ub=np.ones(HORIZON_LENGTH * 2) * VMAX
    )
    
    # Solve optimization (no constraints, penalties in cost function)
    result = minimize(
        cost_function,
        control_init,
        args=(current_state, reference_state, obstacle_predictions),
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 100, 'disp': False}
    )
    
    # Extract first control action
    optimal_control = result.x[:2]
    
    return optimal_control, result.x


def simulate(animate=False):
    """Main simulation loop"""
    print("\n" + "="*70)
    print("   MPC 2D PATH PLANNING - Penalty Method (No Barrier)")
    print("="*70)
    
    # Initialize
    state = np.array([1.5, 1.5])
    goal = np.array([12.0, 12.0])
    
    state_history = np.zeros((NUMBER_OF_TIMESTEPS, 2))
    state_history[0, :] = state
    
    obstacle_trajectories = create_obstacle_trajectories()
    computation_times = []
    previous_control = None
    
    # Setup animation if requested
    if animate:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Simulation loop
    for t in range(NUMBER_OF_TIMESTEPS - 1):
        # Current obstacles at this timestep
        current_obstacles = []
        for i, obs in enumerate(OBSTACLES):
            current_obstacles.append({
                'pos': obstacle_trajectories[t, i, :],
                'vel': obs['vel'],
                'static': obs['static']
            })
        
        # Solve MPC
        start_time = time.time()
        control, previous_control = mpc_controller(state, goal, current_obstacles, previous_control)
        computation_time = time.time() - start_time
        computation_times.append(computation_time)
        
        # Apply control and update state
        state = integrate_state(state, control, TIMESTEP)
        state_history[t + 1, :] = state
        
        # Animation update
        if animate and t % 3 == 0:
            ax.clear()
            ax.set_xlim(0, 14)
            ax.set_ylim(0, 14)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Time: {t*TIMESTEP:.1f}s / {SIM_TIME}s', fontsize=14, fontweight='bold')
            
            # Plot robot
            circle_robot = Circle(state, ROBOT_RADIUS, color='blue', alpha=0.6)
            ax.add_patch(circle_robot)
            ax.plot(state[0], state[1], 'bo', markersize=10, label='Robot')
            
            # Plot trajectory
            ax.plot(state_history[:t+1, 0], state_history[:t+1, 1], 'b-', linewidth=2, alpha=0.4)
            
            # Plot goal
            ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
            
            # Plot obstacles
            colors = ['dimgray', 'dimgray', 'dimgray', 'cyan', 'orange']
            for i, obs in enumerate(OBSTACLES):
                if obs['static']:
                    circle = Circle(obs['pos'], ROBOT_RADIUS + SAFETY_MARGIN, 
                                  color=colors[i], alpha=0.6)
                    ax.add_patch(circle)
                else:
                    circle = Circle(obstacle_trajectories[t, i, :], 
                                  ROBOT_RADIUS + SAFETY_MARGIN, 
                                  color=colors[i], alpha=0.6)
                    ax.add_patch(circle)
                    ax.plot(obstacle_trajectories[:t+1, i, 0], 
                           obstacle_trajectories[:t+1, i, 1], 
                           '--', color=colors[i], alpha=0.4, linewidth=1.5)
            
            ax.legend(loc='upper left')
            plt.pause(0.01)
        
        # Progress
        if (t + 1) % 10 == 0:
            progress = (t + 1) / NUMBER_OF_TIMESTEPS * 100
            print(f"Progress: {progress:.1f}% | State: [{state[0]:.2f}, {state[1]:.2f}] | Time: {computation_time*1000:.1f}ms")
    
    if animate:
        plt.ioff()
        plt.close()
    
    # Final statistics
    print("\n" + "="*70)
    print("   SIMULATION COMPLETE")
    print("="*70)
    
    final_error = goal - state_history[-1, :]
    print(f"\n📍 FINAL POSITION:")
    print(f"   Goal:                    [{goal[0]:.3f}, {goal[1]:.3f}]")
    print(f"   Final:                   [{state_history[-1, 0]:.3f}, {state_history[-1, 1]:.3f}]")
    print(f"   Error:                   [{final_error[0]:.3f}, {final_error[1]:.3f}]")
    print(f"   Error magnitude:         {np.linalg.norm(final_error):.3f} m")
    
    path_length = np.sum(np.linalg.norm(np.diff(state_history, axis=0), axis=1))
    straight_line = np.linalg.norm(goal - state_history[0, :])
    print(f"\n📏 PATH METRICS:")
    print(f"   Path length:             {path_length:.3f} m")
    print(f"   Straight-line distance:  {straight_line:.3f} m")
    print(f"   Path efficiency:         {(straight_line/path_length*100):.1f}%")
    
    total_planning_time = np.sum(computation_times)
    print(f"\n⏱️  COMPUTATION TIME:")
    print(f"   Total planning time:     {total_planning_time:.3f} s")
    print(f"   Average per step:        {np.mean(computation_times)*1000:.2f} ms")
    print(f"   Min per step:            {np.min(computation_times)*1000:.2f} ms")
    print(f"   Max per step:            {np.max(computation_times)*1000:.2f} ms")
    print(f"   Real-time factor:        {(total_planning_time / SIM_TIME):.2f}x")
    
    # Check minimum clearance
    min_distance_overall = float('inf')
    collision_occurred = False
    required_clearance = ROBOT_RADIUS + SAFETY_MARGIN
    
    for t in range(NUMBER_OF_TIMESTEPS):
        for i in range(len(OBSTACLES)):
            distance = np.linalg.norm(state_history[t, :] - obstacle_trajectories[t, i, :])
            min_distance_overall = min(min_distance_overall, distance)
            if distance < required_clearance:
                collision_occurred = True
    
    print(f"\n🚧 COLLISION AVOIDANCE:")
    print(f"   Required clearance:      {required_clearance:.3f} m")
    print(f"   Minimum clearance:       {min_distance_overall:.3f} m")
    if not collision_occurred:
        print(f"   Status:                  ✓ SUCCESS - No collisions!")
    else:
        print(f"   Status:                  ✗ WARNING - Got too close to obstacles!")
    
    print("\n" + "="*70 + "\n")
    
    return state_history, obstacle_trajectories, computation_times


def save_to_excel(state_history, obstacle_trajectories, computation_times, filename='robot_paths_2d_penalty.xlsx'):
    """
    Save robot path and dynamic obstacles paths to Excel file
    """
    # Create time array
    time_array = np.arange(NUMBER_OF_TIMESTEPS) * TIMESTEP
    
    # Create a Pandas Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: Robot Path
        robot_df = pd.DataFrame({
            'Time (s)': time_array,
            'Robot_X (m)': state_history[:, 0],
            'Robot_Y (m)': state_history[:, 1],
            'Planning_Time (ms)': np.array(computation_times) * 1000
        })
        robot_df.to_excel(writer, sheet_name='Robot_Path', index=False)
        
        # Sheet 2: Dynamic Obstacles Only
        dynamic_data = {'Time (s)': time_array}
        dynamic_count = 0
        
        for i, obs in enumerate(OBSTACLES):
            if not obs['static']:  # Only dynamic obstacles
                dynamic_count += 1
                dynamic_label = f"Dynamic_Obs_{dynamic_count}"
                dynamic_data[f'{dynamic_label}_X (m)'] = obstacle_trajectories[:, i, 0]
                dynamic_data[f'{dynamic_label}_Y (m)'] = obstacle_trajectories[:, i, 1]
                dynamic_data[f'{dynamic_label}_Vx (m/s)'] = [obs['vel'][0]] * NUMBER_OF_TIMESTEPS
                dynamic_data[f'{dynamic_label}_Vy (m/s)'] = [obs['vel'][1]] * NUMBER_OF_TIMESTEPS
        
        if dynamic_count > 0:
            dynamic_df = pd.DataFrame(dynamic_data)
            dynamic_df.to_excel(writer, sheet_name='Dynamic_Obstacles', index=False)
        
        # Sheet 3: All Obstacles
        all_obstacles_data = {'Time (s)': time_array}
        
        for i, obs in enumerate(OBSTACLES):
            obs_label = f"Obs_{i+1}"
            obs_type = "Static" if obs['static'] else "Dynamic"
            
            all_obstacles_data[f'{obs_label}_Type'] = [obs_type] * NUMBER_OF_TIMESTEPS
            all_obstacles_data[f'{obs_label}_X (m)'] = obstacle_trajectories[:, i, 0]
            all_obstacles_data[f'{obs_label}_Y (m)'] = obstacle_trajectories[:, i, 1]
        
        all_obs_df = pd.DataFrame(all_obstacles_data)
        all_obs_df.to_excel(writer, sheet_name='All_Obstacles', index=False)
        
        # Sheet 4: Summary Statistics
        goal = np.array([12.0, 12.0])
        initial = np.array([1.5, 1.5])
        final_error = goal - state_history[-1, :]
        
        # Calculate path length
        path_length = np.sum(np.linalg.norm(np.diff(state_history, axis=0), axis=1))
        
        # Calculate minimum distances
        min_distances = []
        for i in range(len(OBSTACLES)):
            min_dist = float('inf')
            for t in range(NUMBER_OF_TIMESTEPS):
                dist = np.linalg.norm(state_history[t, :] - obstacle_trajectories[t, i, :])
                min_dist = min(min_dist, dist)
            min_distances.append(min_dist)
        
        summary_data = {
            'Parameter': [
                'Initial X (m)', 'Initial Y (m)',
                'Goal X (m)', 'Goal Y (m)',
                'Final X (m)', 'Final Y (m)',
                'Final Error X (m)', 'Final Error Y (m)',
                'Total Error Magnitude (m)',
                'Path Length (m)',
                'Straight-line Distance (m)',
                'Path Efficiency (%)',
                'Total Planning Time (s)',
                'Average Planning Time per Step (ms)',
                'Min Planning Time per Step (ms)',
                'Max Planning Time per Step (ms)',
                'Simulation Time (s)',
                'Number of Obstacles',
                'Number of Dynamic Obstacles',
                'Robot Radius (m)',
                'Safety Margin (m)',
                'Required Clearance (m)',
                'Minimum Clearance Achieved (m)'
            ],
            'Value': [
                initial[0], initial[1],
                goal[0], goal[1],
                state_history[-1, 0], state_history[-1, 1],
                final_error[0], final_error[1],
                np.linalg.norm(final_error),
                path_length,
                np.linalg.norm(goal - initial),
                (np.linalg.norm(goal - initial) / path_length * 100),
                np.sum(computation_times),
                np.mean(computation_times) * 1000,
                np.min(computation_times) * 1000,
                np.max(computation_times) * 1000,
                SIM_TIME,
                len(OBSTACLES),
                sum(1 for obs in OBSTACLES if not obs['static']),
                ROBOT_RADIUS,
                SAFETY_MARGIN,
                ROBOT_RADIUS + SAFETY_MARGIN,
                min(min_distances)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✓ Data saved to '{filename}'")
    print(f"  - Robot_Path: Robot trajectory with planning times")
    print(f"  - Dynamic_Obstacles: Paths of dynamic obstacles only")
    print(f"  - All_Obstacles: All obstacle trajectories")
    print(f"  - Summary: Complete statistics and performance metrics")


def plot_results(state_history, obstacle_trajectories):
    """Final visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Trajectory
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 14)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Robot Trajectory (2D) - Penalty Method', fontsize=13, fontweight='bold')
    
    # Plot robot trajectory
    ax1.plot(state_history[:, 0], state_history[:, 1], 'b-', linewidth=3, label='Robot path')
    ax1.plot(state_history[0, 0], state_history[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(12.0, 12.0, 'r*', markersize=20, label='Goal')
    
    # Plot obstacles
    colors = ['dimgray', 'dimgray', 'dimgray', 'cyan', 'orange']
    for i, obs in enumerate(OBSTACLES):
        if obs['static']:
            circle = Circle(obs['pos'], ROBOT_RADIUS + SAFETY_MARGIN, 
                          color=colors[i], alpha=0.6, label=f'Static Obs {i+1}')
            ax1.add_patch(circle)
        else:
            ax1.plot(obstacle_trajectories[:, i, 0], obstacle_trajectories[:, i, 1], 
                    '--', color=colors[i], linewidth=2, alpha=0.6, label=f'Dynamic Obs {i+1}')
            ax1.plot(obstacle_trajectories[0, i, 0], obstacle_trajectories[0, i, 1], 
                    'o', color=colors[i], markersize=8)
            ax1.plot(obstacle_trajectories[-1, i, 0], obstacle_trajectories[-1, i, 1], 
                    's', color=colors[i], markersize=8)
    
    ax1.legend(fontsize=10)
    
    # Plot 2: Errors over time
    time = np.arange(NUMBER_OF_TIMESTEPS) * TIMESTEP
    goal = np.array([12.0, 12.0])
    
    ax2.plot(time, np.abs(goal[0] - state_history[:, 0]), linewidth=2, label='|x_error|')
    ax2.plot(time, np.abs(goal[1] - state_history[:, 1]), linewidth=2, label='|y_error|')
    total_error = np.linalg.norm(goal - state_history, axis=1)
    ax2.plot(time, total_error, linewidth=2, label='Total error', linestyle='--')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Error (m)', fontsize=12)
    ax2.set_title('Tracking Errors', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nmpc_results_2d_penalty.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run simulation with animation
    state_history, obstacle_trajectories, computation_times = simulate(animate=True)
    
    # Save data to Excel
    save_to_excel(state_history, obstacle_trajectories, computation_times, filename='robot_paths_2d_penalty.xlsx')
    
    # Show final plots
    plot_results(state_history, obstacle_trajectories)
    print("Visualization saved to 'nmpc_results_2d_penalty.png'\n")
