import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from collections import deque
import time

# Maze size and settings
maze_size = 15
start_position = (0, 1)
exit_position = (maze_size - 1, maze_size - 2)

# Q-learning hyperparameters
alpha = 0.1        # Learning rate
gamma = 0.9        # Discount factor
epsilon = 0.1      # Exploration rate
num_episodes = 500  # Number of training episodes

# Action space: up, down, left, right
actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
action_size = len(actions)
Q_table = np.zeros((maze_size, maze_size, action_size))  # Initialize Q-table

# Generate a solvable maze
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

    # Ensure entire top row is a black wall except for the start
    maze[0, :] = 1
    maze[start_position] = 0

    return maze

maze = generate_solvable_maze(maze_size)

# Define reward function
def get_reward(position):
    if position == exit_position:
        return 100  # Reward for reaching the goal
    elif maze[position] == 1:
        return -10  # Penalty for hitting a wall
    else:
        return -0.1  # Small penalty for each step

# Check if a move is valid
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
                action_index = random.randint(0, action_size - 1)  # Explore
            else:
                action_index = np.argmax(Q_table[x, y])  # Exploit

            # Take action
            dx, dy = actions[action_index]
            new_x, new_y = x + dx, y + dy
            if is_valid_move(new_x, new_y):
                next_position = (new_x, new_y)
            else:
                next_position = position

            # Get reward
            reward = get_reward(next_position)

            # Q-learning update
            old_value = Q_table[x, y, action_index]
            next_max = np.max(Q_table[new_x, new_y])
            Q_table[x, y, action_index] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            # Move to the next position
            position = next_position

# Train the agent
train_agent()

# Visualization setup
cmap = colors.ListedColormap(['white', 'black', 'blue'])
norm = colors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)
fig, ax = plt.subplots()
img = ax.imshow(maze, cmap=cmap, norm=norm, interpolation='nearest')
object_position = list(start_position)
path_stack = [start_position]
start_time = time.time()

# Animation for showing the learned path
def update(frame):
    global object_position
    x, y = object_position

    # Stop the animation if the agent reaches the exit
    if object_position == exit_position:
        ani.event_source.stop()
        end_time = time.time()
        print(f"Maze completed in {end_time - start_time:.2f} seconds")
    else:
        action_index = np.argmax(Q_table[x, y])  # Choose best learned action
        dx, dy = actions[action_index]
        new_x, new_y = x + dx, y + dy
        if is_valid_move(new_x, new_y):
            object_position = (new_x, new_y)
            maze[new_x, new_y] = 3  # Mark visited path for visualization

        img.set_data(maze)
    return [img]

ani = FuncAnimation(fig, update, frames=200, blit=True, repeat=False)
plt.show()
