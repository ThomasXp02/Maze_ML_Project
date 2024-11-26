import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from collections import deque
import time  # Import time module

# Maze size and settings
maze_size = 15
start_position = (0, 1)  # Start at the top edge of the maze
exit_position = (maze_size - 1, maze_size - 2)  # Exit at the bottom edge of the maze

def generate_solvable_maze(size):
    # Initialize maze with walls (1s) and paths (0s)
    maze = np.ones((size, size))
    maze[start_position] = 0
    maze[exit_position] = 0

    # Recursive backtracking to create a perfect maze
    def carve_path(x, y):
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx, ny] == 1:
                maze[nx, ny] = 0
                maze[x + dx // 2, y + dy // 2] = 0  # Remove wall between cells
                carve_path(nx, ny)

    carve_path(1, 1)  # Start carving from (1,1) for a better maze structure

    # Add dead ends to make the maze more complex
    def add_dead_ends():
        for _ in range(size * 2):  # Increase or decrease for more/less dead ends
            x, y = random.randint(1, size - 2), random.randint(1, size - 2)
            if maze[x, y] == 1:  # Only start carving from walls
                directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
                random.shuffle(directions)
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 1 <= nx < size - 1 and 1 <= ny < size - 1 and maze[nx, ny] == 0:
                        maze[x, y] = 0
                        maze[x + dx // 2, y + dy // 2] = 0  # Create a dead-end branch
                        break

    add_dead_ends()

    # Check if exit is reachable from start using BFS
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

    # If the exit is not reachable, carve a direct path to the exit
    if not is_exit_reachable():
        x, y = exit_position
        while (x, y) != start_position:
            maze[x, y] = 0  # Carve path
            if x > start_position[0]:
                x -= 1
            elif y > start_position[1]:
                y -= 1

    # Ensure the entire top row is a black wall except for the start
    maze[0, :] = 1
    maze[start_position] = 0  # Keep the start open

    return maze

maze = generate_solvable_maze(maze_size)

# Define colors for walls, paths, and current position
cmap = colors.ListedColormap(['white', 'black', 'blue'])
norm = colors.BoundaryNorm([0, 0.5, 1.5, 2.5], cmap.N)

# Set start and end positions explicitly in maze
maze[start_position] = 0  # Path for start
maze[exit_position] = 0   # Path for exit

# Animation setup
fig, ax = plt.subplots()
img = ax.imshow(maze, cmap=cmap, norm=norm, interpolation='nearest')
object_position = list(start_position)
visited_positions = set()
path_stack = [start_position]  # Stack to track the path

# Start time for the maze traversal
start_time = time.time()

# Movement logic for navigating through the maze
def valid_moves(x, y):
    moves = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze_size and 0 <= ny < maze_size and maze[nx, ny] == 0 and (nx, ny) not in visited_positions:
            moves.append((nx, ny))
    return moves

def update(frame):
    global object_position, path_stack
    x, y = object_position

    # Stop the animation and print elapsed time when the object reaches the exit
    if object_position == exit_position:
        ani.event_source.stop()  # Stop animation when reaching the exit
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Maze completed in {elapsed_time:.2f} seconds")
    else:
        moves = valid_moves(x, y)
        if moves:
            # Move to a new cell if there are valid moves
            next_position = random.choice(moves)
            visited_positions.add(next_position)
            path_stack.append(object_position)  # Push current position to the stack
            object_position = next_position
            maze[object_position] = 3  # Mark current position with blue
        else:
            # Dead end: backtrack to the last position in the stack
            if path_stack:
                object_position = path_stack.pop()  # Pop last position from the stack

        img.set_data(maze)
    return [img]

ani = FuncAnimation(fig, update, frames=200, blit=True, repeat=False)
plt.show()
