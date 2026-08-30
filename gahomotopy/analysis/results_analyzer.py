import json
from gahomotopy.kinematics.roarm import ROARM3DOF
import numpy as np
import math

def load_json_file(file_name):
    """
    Loads and returns the data from a JSON file.
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return f"Error: The file '{file_name}' was not found."
    except json.JSONDecodeError:
        return "Error: Failed to decode JSON. Check if the file format is valid."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def get_time(data):
    planning_time = data["PlanningTime"]
    return planning_time

def get_distance(data):
    all_positions = [point["positions"][:3] for point in data["trajectory"]]

    arm=ROARM3DOF()

    pos_len=len(all_positions)

    pos_array = np.zeros((pos_len, 3))
    counter=0
    for pos in all_positions:
        pos_space=arm.directKinematics(pos[0],pos[1],pos[2])
        pos_array[counter,0]=math.degrees(pos_space[0,3])
        pos_array[counter,1]=math.degrees(pos_space[1,3])
        pos_array[counter,2]=math.degrees(pos_space[2,3])
        counter+=1

    diffs = np.diff(pos_array, axis=0)

    # 2. Calculate the Euclidean distance for each segment
    # np.linalg.norm calculates the magnitude of the vectors
    segment_distances = np.linalg.norm(diffs, axis=1)

    # 3. Sum them up to get the total path length
    total_distance = np.sum(segment_distances)
    total_distance_cm=total_distance/10

    
    return total_distance_cm

# Example usage:
file_name="moveit_results/obstacles1_trajectory_output.json"
file_names=["moveit_results/obstacles1_trajectory_output.json","moveit_results/obstacles3_trajectory_output.json","moveit_results/obstacles4_trajectory_output.json","moveit_results/obstacles5_trajectory_output.json","moveit_results/obstacles6_trajectory_output.json","moveit_results/obstacles8_trajectory_output.json",]

for file_name in file_names:
    my_data = load_json_file(file_name)
    plannig_time=get_time(my_data)
    total_distance_cm=get_distance(my_data)
    print(file_name)
    print(f"Planning Time: {plannig_time:.4f} s")
    print(f"Total distance: {total_distance_cm:.4f} cm")
#print(my_data)


