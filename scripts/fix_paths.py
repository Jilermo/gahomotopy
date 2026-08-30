import os
import numpy as np

def extend_path_to_goal(file_path, goal_joint_state, num_waypoints=20):
    """
    Reads an n x 7 trajectory file, interpolates from the final point to a goal state,
    and saves the extended trajectory.
    
    :param file_path: str, path to the original .npy file
    :param goal_joint_state: list or array of 6 elements (the target DoF configurations)
    :param num_waypoints: int, how many steps to generate from the last point to the goal
    """
    # 1. Load the original trajectory
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")
        
    original_path = np.load(file_path)
    print(f"Loaded original path with shape: {original_path.shape}")
    
    # Ensure it matches the expected n x 7 structure
    if original_path.ndim != 2 or original_path.shape[1] != 7:
        raise ValueError(f"Expected an (n x 7) array, but got shape {original_path.shape}")
        
    if len(goal_joint_state) != 6:
        raise ValueError(f"Goal state must have exactly 6 joint values, got {len(goal_joint_state)}")

    # 2. Extract the last point in the current trajectory
    last_row = original_path[-1, :]
    start_joint_state = last_row[:6]
    last_7th_val = last_row[6] # Kept to maintain the 7th column structure

    # 3. Generate the interpolated waypoints for the 6 DoF
    # np.linspace generates smooth steps between the start and goal states
    interpolated_joints = np.linspace(start_joint_state, goal_joint_state, num_waypoints + 1)
    
    # np.linspace includes the start point, so we slice it out to avoid duplicating the last row
    new_joints = interpolated_joints[1:]
    
    # 4. Reconstruct the 7th column for the new waypoints
    # We match the length of the new points and fill it with the last known 7th column value
    new_7th_col = np.full((new_joints.shape[0], 1), last_7th_val)
    
    # Combine the new joints and the 7th column together
    extension = np.hstack((new_joints, new_7th_col))
    
    # 5. Append the extension to the original path
    fixed_path = np.vstack((original_path, extension))
    
    # 6. Save the new file with '_fixed' appended to the name
    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_fixed{ext}"
    
    np.save(output_path, fixed_path)
    print(f"Successfully extended path! New shape: {fixed_path.shape}")
    print(f"Saved fixed path to: {output_path}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define a dummy file for testing purposes
    test_file = "/home/jilermo/Documents/MIA/ur_ws/results/Manual/obstacles8.npy"
    
    
    # --- Running the fix ---
    # Define your target 6 DoF configuration
    my_goal = [4.92, -139.77, -109.72, 65.64, 88.94, 89.08] 
    steps_to_add = 350
    
    extend_path_to_goal(
        file_path=test_file, 
        goal_joint_state=my_goal, 
        num_waypoints=steps_to_add
    )
    
    # Clean up mock base file if you want, or replace 'trajectory.npy' with your real path