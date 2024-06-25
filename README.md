# Smart-Grid
Agent explores 5x5 grid, learning best moves (Q-learning) to reach goal. Remembers past experiences (replay) for faster, smarter training. Visualizes learning process.

Imports:

torch: This library is used for deep learning tasks. In this case, it's likely not directly used for the Q-learning algorithm but might be used for a neural network implementation (not included here).

pygame: This library is used for creating a graphical user interface (GUI) to visualize the agent's movement on a grid.

random: This library provides functions for generating random numbers, used for exploration during training.

time: This library allows measuring time for tracking the learning progress.

Initialization:

Device: Checks if a GPU is available and sets the device for tensor computations (GPU for faster training if available).
Environment: Defines parameters for the grid world: number of states (grid size), number of actions (movement options), goal state, and start state.
Learning Parameters: Sets the learning rate (how much the agent updates its knowledge based on rewards), discount factor (importance of future rewards), exploration rate (how often the agent takes random actions), and other hyperparameters for the algorithm.
Experience Replay: Initializes an empty list memory to store past experiences and sets its maximum size.
Q-table: Creates a Q-table (Q) with dimensions matching the number of states and actions. The Q-table will store the estimated Q-values for each state-action pair.

Pygame Setup:

Initializes Pygame for visualization.
Defines the grid size and creates a screen for displaying the environment.
Sets a clock to limit the frame rate for smoother visuals.
Creates a font for displaying text on the screen.

Helper Functions:

state_to_position - Converts a state number (index in the Q-table) to its corresponding position (x, y coordinates) on the grid.
step - Takes a state and an action as input and returns the next state and the reward received for taking that action in the environment.
Experience Replay Function:

experience_replay - This function samples a batch of experiences from the memory and updates the Q-values in the Q-table. It uses the Bellman equation with experience replay to learn from past experiences, making the training process more efficient and stable.
Q-learning Algorithm with Experience Replay:

Main Loop: Runs for a specified number of episodes (training iterations).
Episode Loop: Runs until the agent reaches the goal state or a maximum number of steps is reached within an episode.
Selects an action based on the epsilon-greedy strategy: explore randomly with some probability (epsilon) or choose the action with the highest Q-value.
Takes the action, observes the next state and reward.
Stores the experience (state, action, reward, next state) in the memory.
Updates the state for the next step.
Visualizes the environment using Pygame (drawing the grid, agent position, and other information).
Decay Epsilon: Gradually reduces the exploration rate (epsilon) over time, encouraging the agent to exploit its learned knowledge more as training progresses.
Experience Replay: Calls the experience_replay function to update the Q-table based on sampled experiences from the memory.
Performance Tracking: Calculates and prints the average number of moves taken to reach the goal state over the last 10 episodes.
Finalization:

Quits Pygame, cleaning up resources used for visualization.
Overall, this code implements a Q-learning algorithm with experience replay for an agent navigating a grid world to reach a goal state. It also utilizes Pygame to visualize the training process.
