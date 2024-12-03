import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from collections import deque
import time  # Import time module
import pybullet as p
import pybullet_data
import math

# Maze size and settings
maze_size = 14  # Updated maze size to 14x14
start_position = (0, 1)  # Start at (0, 1) on the top edge of the maze
exit_position = (maze_size - 1, maze_size - 2)  # Exit at the bottom edge of the maze

def generate_solvable_maze(size):
    maze = np.ones((size, size), dtype=int)  # Use 1 for walls and 0 for paths
    maze[start_position] = 0
    maze[exit_position] = 0

    def carve_path(x, y):
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx, ny] == 1:
                maze[nx, ny] = 0
                maze[x + dx // 2, y + dy // 2] = 0
                carve_path(nx, ny)

    carve_path(1, 1)

    def add_dead_ends():
        for _ in range(size * 2):
            x, y = random.randint(1, size - 2), random.randint(1, size - 2)
            if maze[x, y] == 1:
                directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx, ny] == 0:
                        maze[x, y] = 0
                        maze[x + dx // 2, y + dy // 2] = 0
                        break

    add_dead_ends()

    def is_exit_reachable():
        queue = deque([start_position])
        visited = set([start_position])
        while queue:
            x, y = queue.popleft()
            if (x, y) == exit_position:
                return True
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    if not is_exit_reachable():
        x, y = exit_position
        while (x, y) != start_position:
            maze[x, y] = 0
            if x > start_position[0]:
                x -= 1
            elif y > start_position[1]:
                y -= 1

    maze[0, :] = 1
    maze[start_position] = 0

    return maze

maze = generate_solvable_maze(maze_size)

# Matplotlib 2D visualization setup
cmap = colors.ListedColormap(['white', 'black'])  # White for path (0), black for walls (1)
norm = colors.BoundaryNorm([0, 0.5, 1.5], cmap.N)  # 0: path, 1: wall
fig, ax = plt.subplots()
img = ax.imshow(maze, cmap=cmap, norm=norm, interpolation='nearest')

# Matplotlib 2D visualization
plt.show()  # Display the 2D maze before starting PyBullet

# PyBullet 3D simulation setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Adjust floor size for 14x14 maze
floor_size = 14
floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[floor_size, floor_size, 0.1])
floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[floor_size, floor_size, 0.1], rgbaColor=(0.5, 0.5, 0.5, 1))
floor_id = p.createMultiBody(baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, 0])

# Wall creation function in PyBullet with expanded walls to touch each other
def create_wall(x, y, color=(0.6, 0.3, 0.0, 1)):
    # Create a vertical wall at (x, y) with no gaps between adjacent walls
    wall_height = 0.5  # Wall height
    wall_width = 1  # Expanded wall width to touch adjacent walls
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_width / 2, wall_width / 2, wall_height])
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_width / 2, wall_width / 2, wall_height], rgbaColor=color)
    p.createMultiBody(baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[x, y, wall_height / 2])

# Create walls based on the maze
wall_thickness = 0.1
for i in range(maze_size):
    for j in range(maze_size):
        if maze[i, j] == 1:
            # Adjust the position so the start position is at the origin (0, 0)
            create_wall(j - start_position[1], start_position[0] - i)

# Camera setup for 3D visualization
p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[start_position[0], start_position[1], 0])

# Set the start position in PyBullet (converted to meters)
start_pos = start_position  # Start at (0, 1) in the grid
start_position_3d = [start_pos[1], -start_pos[0], 0.5]  # Convert grid to PyBullet coordinates: (x, -y, z)
robot_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
robot_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=(0, 0, 1, 1))  # Blue sphere as the robot
robot_id = p.createMultiBody(baseCollisionShapeIndex=robot_collision_shape, baseVisualShapeIndex=robot_visual_shape, basePosition=start_position_3d)

# Run both visualizations
while True:
    p.stepSimulation()
    time.sleep(1. / 240.)
