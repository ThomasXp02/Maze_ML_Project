# AI Maze Generation and Visualization
This python code is programed to randomly create a maze, which is solved side by side in to ways: random movements and machine learning. These two patterns are output in a 2-D gif animating their path through the same randomized maze, with PyBullet visualizing the machine learning version's path through the same maze.

![maze_navigation_comparison3](https://github.com/user-attachments/assets/f0ea7744-c3eb-4527-837d-aee01ca043f2) 
![M3_Py](https://github.com/user-attachments/assets/d9e11b0e-27ad-4dc1-819f-03c536ebae23)

## Team Members
1. Thom Pham
2. Jacob Grossman
3. Sydni Cobb

## Package Installation Requirements (Windows)
Due to the visualization aspects of this code, 2 packages will need to be installed outside of what are likely are being installed:

```
pip install pybullet
pip install matplotlib
```
This assumes that `numpy`, `random`, `time`, and `collections` are installed. 

## Outcomes
There are 3 outputs from `Integrated_ML_PyBullet.py`:
1. A side-by-side gif of a randomized maze with the random and machine learning paths.
2. A console output tracking the times each version take to complete the maze.
3. PyBullet visualization of the machine learning version of the maze in PyBullet's GUI.

## Some intro sources to help with machine learning algos
- https://github.com/duncantmiller/ai-developer-resources
- https://techvidvan.com/tutorials/reinforcement-learning/
- https://github.com/bulletphysics/bullet3/tree/master
- https://www.datacamp.com/tutorial/bellman-equation-reinforcement-learning
- https://www.kdnuggets.com/2022/06/beginner-guide-q-learning.html
- https://www.geeksforgeeks.org/q-learning-in-python/#
