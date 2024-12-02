import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import time

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

def train_agent():
    for episode in range(num_episodes):
        position = start_position
        while position != exit_position:
            x, y = position
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, action_size - 1)
            else:
                action_index = np.argmax(Q_table[x, y])
            dx, dy = actions[action_index]
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < maze_size and 0 <= new_y < maze_size and maze[new_x, new_y] == 0:
                next_position = (new_x, new_y)
            else:
                next_position = position
            reward = 100 if next_position == exit_position else -0.1
            old_value = Q_table[x, y, action_index]
            next_max = np.max(Q_table[new_x, new_y])
            Q_table[x, y, action_index] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            position = next_position

train_agent()

# Visualization setup
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
            if 0 <= x+dx < maze_size and 0 <= y+dy < maze_size and mazes[idx][x+dx, y+dy] == 0:
                object_positions[idx] = [x+dx, y+dy]

        # Update the maze visualization
        nx, ny = object_positions[idx]
        mazes[idx][x, y] = 0  # Reset current position
        mazes[idx][nx, ny] = 3  # Mark new position
        imgs[idx].set_data(mazes[idx])

    return imgs

ani = FuncAnimation(fig, update, frames=200, blit=True, repeat=False)
plt.tight_layout()
plt.show()
