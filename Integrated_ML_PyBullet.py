import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import time
import pybullet as p
import pybullet_data
from collections import deque

# Maze size and settings
maze_size = 15
start_position = (0, 1)
exit_position = (maze_size - 1, maze_size - 2)

# Define colors for walls, paths, and current position
cmap = colors.ListedColormap(['white', 'black', 'blue'])
norm = colors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)

# Common maze generation
def generate_solvable_maze(size):
    maze = np.ones((size, size))
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

# Q-learning setup
alpha, gamma, epsilon = 0.1, 0.9, 0.1
num_episodes = 500
actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
action_size = len(actions)
Q_table = np.zeros((maze_size, maze_size, action_size))

# Helper function: Calculate reward
def get_reward(position):
    if position == exit_position:
        return 100
    elif maze[position] == 1:
        return -10
    else:
        return -0.1

# Helper function: Check if a move is valid
def is_valid_move(x, y):
    return 0 <= x < maze_size and 0 <= y < maze_size and maze[x, y] == 0

# Q-learning training function
def train_agent():
    for episode in range(num_episodes):
        position = start_position
        while position != exit_position:
            x, y = position

            # Choose action using epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, action_size - 1)
            else:
                action_index = np.argmax(Q_table[x, y])

            dx, dy = actions[action_index]
            new_x, new_y = x + dx, y + dy
            if is_valid_move(new_x, new_y):
                next_position = (new_x, new_y)
            else:
                next_position = position

            reward = get_reward(next_position)
            old_value = Q_table[x, y, action_index]
            next_max = np.max(Q_table[new_x, new_y])
            Q_table[x, y, action_index] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            position = next_position

# Train the agent
train_agent()

# Visualization setup remains unchanged
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
mazes = [maze.copy(), maze.copy()]
imgs = [axes[0].imshow(mazes[0], cmap=cmap, norm=norm),
        axes[1].imshow(mazes[1], cmap=cmap, norm=norm)]
axes[0].set_title("Random Navigation")
axes[1].set_title("Q-Learning Navigation")

object_positions = [list(start_position), list(start_position)]
path_stacks = [[start_position], []]  # Path stack for random navigation
visited = [set(), set()]  # Visited cells for each algorithm

start_times = [time.time(), time.time()]
end_times = [None, None]  # To record the completion time for each algorithm

def update(frame):
    global end_times
    for idx in range(2):
        # Skip updates for completed algorithms
        if end_times[idx] is not None:
            continue

        x, y = object_positions[idx]

        # Stop and record the time if the agent reaches the exit
        if (x, y) == exit_position:
            end_times[idx] = time.time()
            elapsed_time = end_times[idx] - start_times[idx]
            print(f"Algorithm {idx+1} completed in {elapsed_time:.2f} seconds")
            continue

        if idx == 0:  # Random navigation with backtracking
            moves = [(dx, dy) for dx, dy in actions if
                     0 <= x+dx < maze_size and 0 <= y+dy < maze_size and mazes[idx][x+dx, y+dy] == 0 and (x+dx, y+dy) not in visited[idx]]
            if moves:
                dx, dy = random.choice(moves)
                nx, ny = x + dx, y + dy
                visited[idx].add((nx, ny))
                path_stacks[idx].append((nx, ny))
                object_positions[idx] = [nx, ny]
            elif path_stacks[idx]:
                nx, ny = path_stacks[idx].pop()
                object_positions[idx] = [nx, ny]
        else:  # Q-learning
            action_index = np.argmax(Q_table[x, y])
            dx, dy = actions[action_index]
            if is_valid_move(x + dx, y + dy):
                object_positions[idx] = [x + dx, y + dy]

        # Update the maze visualization
        nx, ny = object_positions[idx]
        mazes[idx][x, y] = 0  # Reset current position
        mazes[idx][nx, ny] = 3  # Mark new position
        imgs[idx].set_data(mazes[idx])

    return imgs

ani = FuncAnimation(fig, update, frames=200, blit=True, repeat=False)

# Save the animation as a GIF
ani.save('maze_navigation.gif', writer='pillow', fps=10)
plt.tight_layout()
plt.show()

# PyBullet 3D simulation setup
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

floor_size = 15
floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[floor_size, floor_size, 0.1])
floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[floor_size, floor_size, 0.1], rgbaColor=(0.5, 0.5, 0.5, 1))
p.createMultiBody(baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, 0])

def create_wall(x, y, color=(0.6, 0.3, 0.0, 1)):
    wall_height = 1.0  # Increased height to block robot
    wall_width = 1
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_width / 2, wall_width / 2, wall_height])
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_width / 2, wall_width / 2, wall_height], rgbaColor=color)
    p.createMultiBody(baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=[x, y, wall_height / 2])

for i in range(maze_size):
    for j in range(maze_size):
        if maze[i, j] == 1:
            create_wall(j - start_position[1] + 1, start_position[0] - i)

p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[start_position[0], start_position[1], 0])

start_pos = start_position
start_position_3d = [start_pos[1] - 1, -start_pos[0] + 1, 0.5]
robot_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
robot_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=(0, 0, 1, 1))
robot_id = p.createMultiBody(baseCollisionShapeIndex=robot_collision_shape, baseVisualShapeIndex=robot_visual_shape, basePosition=start_position_3d)

def move_robot(position, target, speed=0.5):
    steps = int(1 / speed)
    x_diff = (target[0] - position[0]) / steps
    y_diff = (target[1] - position[1]) / steps

    for _ in range(steps):
        position[0] += x_diff
        position[1] += y_diff
        position[2] = 0.5
        p.resetBasePositionAndOrientation(robot_id, position, [0, 0, 0, 1])
        p.stepSimulation()
        time.sleep(1. / 240.)

# Infinite loop for continuous animation
while True:
    position = start_position
    robot_pos_3d = start_position_3d

    while position != exit_position:
        x, y = position
        action_index = np.argmax(Q_table[x, y])
        dx, dy = actions[action_index]
        new_x, new_y = x + dx, y + dy

        if is_valid_move(new_x, new_y):
            next_position = (new_x, new_y)
            next_position_3d = [next_position[1], -next_position[0], 0.5]
            move_robot(robot_pos_3d, next_position_3d, speed=0.1)
            position = next_position
            robot_pos_3d = next_position_3d
        else:
            next_position_3d = robot_pos_3d
            p.resetBasePositionAndOrientation(robot_id, next_position_3d, [0, 0, 0, 1])

        p.stepSimulation()
        time.sleep(0.05)

    # Pause briefly at the exit before resetting
    time.sleep(1)

    # Reset robot's position back to the start
    p.resetBasePositionAndOrientation(robot_id, start_position_3d, [0, 0, 0, 1])
    p.stepSimulation()
    time.sleep(0.5)
