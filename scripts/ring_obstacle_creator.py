import math

def generate_ring_obstacles(center, ring_radius, fixed_axis, obstacle_radius, num_points=16):
    """
    Generates a list of obstacle dictionaries approximating a ring shape.
    
    :param center: tuple (x, y, z) representing the center of the ring
    :param ring_radius: float, the radius of the overall ring
    :param fixed_axis: str, 'x', 'y', or 'z' (the axis perpendicular to the ring's plane)
    :param obstacle_radius: float, the radius assigned to each individual obstacle point
    :param num_points: int, how many obstacles to use to approximate the circle
    :return: list of dicts in the specified obstacle format
    """
    cx, cy, cz = center
    obstacles = []
    
    fixed_axis = fixed_axis.lower()
    if fixed_axis not in ['x', 'y', 'z']:
        raise ValueError("fixed_axis must be 'x', 'y', or 'z'")

    for i in range(num_points):
        # Calculate the angle around the circle (in radians)
        angle = (2 * math.pi * i) / num_points
        
        # Calculate offsets based on the ring radius
        offset_1 = ring_radius * math.cos(angle)
        offset_2 = ring_radius * math.sin(angle)
        
        # Assign coordinates depending on which axis is fixed
        if fixed_axis == 'x':
            # Ring lies in the YZ plane (moves on y and z)
            x = cx
            y = cy + offset_1
            z = cz + offset_2
        elif fixed_axis == 'y':
            # Ring lies in the XZ plane (moves on x and z)
            x = cx + offset_1
            y = cy
            z = cz + offset_2
        else: # 'z'
            # Ring lies in the XY plane (moves on x and y)
            x = cx + offset_1
            y = cy + offset_2
            z = cz

        # Round values for cleaner formatting
        point = (round(x, 2), round(y, 2), round(z, 2))
        
        obstacles.append({
            'center': point,
            'radius': obstacle_radius
        })
        
    return obstacles

# --- Example Usage ---
if __name__ == "__main__":
    # Your example: Center (0,0,0), Ring Radius 20, fixed on X axis, obstacle radius 5
    ring_obstacles = generate_ring_obstacles(
        center=(330.0, -107.0, 185.0),
        ring_radius=110,
        fixed_axis='x',
        obstacle_radius=70,
        num_points=10  # Kept small for easy reading in the output
    )
    
    print("obstacles = [")
    for obs in ring_obstacles:
        print(f"    {obs},")
    print("]")