import numpy as np
import matplotlib.pyplot as plt
import time

class ArmVisualizer:
    def __init__(self, obstacles=None):
        # 1. Turn on interactive mode for non-blocking updates
        plt.ion() 
        
        # 2. Set up the figure and 3D projection
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        
        # 3. Initialize an empty line for the arm joints/links
        self.line, = self.ax.plot([], [], [], 'o-', lw=3, markersize=8, color='blue')
        
        # 4. Set static limits (matching your updated mm dimensions)
        self.ax.set_xlim3d([-300, 300])
        self.ax.set_ylim3d([-200, 300])
        self.ax.set_zlim3d([0, 600])

        self.ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
        
        # Labels for clarity
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title('Robotic Arm Path Visualizer')

        # 5. Draw obstacles immediately if they are passed during creation
        if obstacles is not None:
            self.draw_obstacles(obstacles)

    def draw_obstacles(self, obstacles):
        """
        Plots static spherical obstacles.
        obstacles: a numpy array of shape (M, 4) -> [x, y, z, radius]
        """
        # Create standard mathematical sphere coordinates
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        
        # Standard parametric sphere equation components
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

        for obs in obstacles:
            ox, oy, oz, radius = obs
            
            # Scale and translate the base sphere coordinates to match the obstacle
            x = radius * x_sphere + ox
            y = radius * y_sphere + oy
            z = radius * z_sphere + oz
            
            # Plot the sphere as a surface. Alpha makes it semi-transparent so you can 
            # see if the arm line cuts through it.
            self.ax.plot_surface(x, y, z, color='red', alpha=0.1, edgecolor='none')


    def update(self, matrices):
        """
        Takes an (N, 4, 4) numpy array, extracts positions, 
        and updates the arm plot without blocking.
        """
        x_data = matrices[:, 0, 3]
        y_data = matrices[:, 1, 3]
        z_data = matrices[:, 2, 3]

        # Update the line data efficiently (leaves static obstacles alone)
        self.line.set_data(x_data, y_data)
        self.line.set_3d_properties(z_data)
        
        # Flush the GUI events to redraw the screen
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def drawArm(arm, ang, visualizer):
    poss = arm.getMidPosFast(ang)
    visualizer.update(poss)

def animatePath(file,arm,obstacles,start,goal,name):
    

    # 1. Load your trajectory steps
    data = np.load(file)

    arm.setObstaclesPos(obstacles)

    # 2. Extract the static obstacles from the arm object
    obstacles = arm.obstacles

    visualizer = ArmVisualizer(obstacles=obstacles)


    # This is a test to see changes saved
    # 4. Loop through the path array
    counter=0
    for i in data:
        
        # Assuming your data row contains configurations/angles
        # Note: In your snippet you pass `i` down to drawArm instead of `ang` 
        # Make sure your arm object is getting what it expects!
        if counter%20==0:
            drawArm(arm, i, visualizer)

        counter+=1
        
        # Optional: Add a tiny pause so the human eye can track it
        # Adjust 0.05 to speed up or slow down the playback
        #time.sleep(0.05)

    print("Finished animation")
    plt.ioff() # Turn off interactive mode so the final frame stays up
    plt.show() # Block here at the very end so the program doesn't instantly close


def visualizeTrajectoryPositions(arm,obstacles,start,goal,name):
    arm.setObstaclesPos(obstacles)

    # 2. Extract the static obstacles from the arm object
    obstacles = arm.obstacles

    visualizer = ArmVisualizer(obstacles=obstacles)

    drawArm(arm, start, visualizer)
    input("Press any key to continue")

    drawArm(arm, goal, visualizer)
    input("Press any key to continue")


def visualizeArm(arm):
    #Draw arm in one position
    ang=[0,-90,0,0,0,0]
    visualizer = ArmVisualizer()
    drawArm(arm,ang,visualizer)

    plt.ioff() # Turn off interactive mode so the final frame stays up
    plt.show() # Block here at the very end so the program doesn't instantly close

    


if __name__ == "__main__":    
    from gahomotopy.kinematics.ur3e import UR3E
    #visualizer = ArmVisualizer()

    arm=UR3E()

    from gahomotopy.planning.experiments import LoadScenario

    obstacles,start,goal,name=LoadScenario("obstacles2")

    #Animate a path that is already save in a file
    #animatePath("src/ur_homotopy/ur_homotopy/obstacles1.npy",arm)
    #animatePath("results/Automatic/obstacles4.npy",arm,obstacles,start,goal,name)

    #See one position of the arm
    #visualizeArm(arm)

    #Visualize starting, ending and obstacles of specific scenarios
    visualizeTrajectoryPositions(arm,obstacles,start,goal,name)
    