
import torch
import pygame
import random
import time

torch.cuda.empty_cache()  # Clear GPU cache

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_states = 25  # 5x5 grid
n_actions = 8  # Up, Down, Left, Right, Up-Left, Up-Right, Down-Left, Down-Right
goal_state = 24  # Top-right corner of the grid
start_state = 0  # Bottom-left corner of the grid
learning_rate = 0.5
discount_factor = 0.95
epsilon_start = 1.0  # Initial exploration rate
epsilon_end = 0.1  # Final exploration rate (minimum)
epsilon_decay = 0.995  # Decay rate for exploration rate
batch_size = 256
memory_size = 100000
memory = []
Q = torch.zeros(n_states, n_actions, device=device)

# Pygame initialization
pygame.init()
grid_size = 100
screen = pygame.display.set_mode((5 * grid_size, 5 * grid_size))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 18)

def state_to_position(state):
    """Converts a state number to a position on the grid."""
    return state % 5 * grid_size, state // 5 * grid_size

def step(state, action):
    """Returns the next state and reward given an action."""
    x, y = state_to_position(state)
    if action == 0:  # Up
        x = max(0, x - grid_size)
    elif action == 1:  # Down
        x = min(4 * grid_size, x + grid_size)
    elif action == 2:  # Left
        y = max(0, y - grid_size)
    elif action == 3:  # Right
        y = min(4 * grid_size, y + grid_size)
    elif action == 4:  # Up-Left
        x = max(0, x - grid_size)
        y = max(0, y - grid_size)
    elif action == 5:  # Up-Right
        x = max(0, x - grid_size)
        y = min(4 * grid_size, y + grid_size)
    elif action == 6:  # Down-Left
        x = min(4 * grid_size, x + grid_size)
        y = max(0, y - grid_size)
    elif action == 7:  # Down-Right
        x = min(4 * grid_size, x + grid_size)
        y = min(4 * grid_size, y + grid_size)
    next_state = x // grid_size + y // grid_size * 5
    reward = 1 if next_state == goal_state else 0
    return next_state, reward

# Experience replay function
def experience_replay():
    """Sample a batch of experiences from memory and update Q-values."""
    batch = random.sample(memory, min(len(memory), batch_size))
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(states, device=device)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, device=device)
    next_states = torch.tensor(next_states, device=device)
    dones = torch.tensor(dones, device=device, dtype=torch.float32)  # Convert to float tensor

    max_next_Q = torch.max(Q[next_states], dim=1)[0]
    targets = rewards + discount_factor * (1.0 - dones) * max_next_Q  # Use float tensor

    Q_values = Q[states, actions]
    Q[states, actions] += learning_rate * (targets - Q_values)

# Q-learning algorithm with experience replay
total_episodes = 1000
epsilon = epsilon_start
start_time = time.time()  # Start time for learning
total_moves = 0  # Total moves for calculating average
episodes_since_average = 0  # Count of episodes since last average update
average_moves = 0  # Average moves to goal destination
for episode in range(total_episodes):
    state = start_state
    done = False
    move_count = 0  # Track total move count

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = torch.argmax(Q[torch.tensor(state, device=device)])

        next_state, reward = step(state, action)
        memory.append((state, action, reward, next_state, False))  # Store experience
        if len(memory) > memory_size:
            del memory[0]  # Remove oldest experience if memory buffer is full

        state = next_state
        move_count += 1  # Increment move count
        total_moves += 1  # Increment total moves
        if state == goal_state:
            done = True

        screen.fill((255, 255, 255))
        # Draw grid lines
        for i in range(5):
            for j in range(5):
                pygame.draw.rect(screen, (0, 0, 0), (i * grid_size, j * grid_size, grid_size, grid_size), 1)
        # Draw goal state
        pygame.draw.rect(screen, (144, 238, 144), (4 * grid_size, 4 * grid_size, grid_size, grid_size)) # Light green color
        # Draw agent state
        pygame.draw.rect(screen, (173, 216, 230), (state_to_position(state)[0]+20, state_to_position(state)[1]+20, grid_size-40, grid_size-40)) # Light blue color

        # Display move count, memory utilization, epsilon, and learning time
        move_text = font.render(f"Moves: {move_count}", True, (0, 0, 0))
        screen.blit(move_text, (10, 10 + 20 * 0)) # Adjusted Y coordinate
        memory_text = font.render(f"Memory: {len(memory)}/{memory_size}", True, (0, 0, 0))
        screen.blit(memory_text, (10, 10 + 20 * 1)) # Adjusted Y coordinate
        epsilon_text = font.render(f"Epsilon: {epsilon:.2f}", True, (0, 0, 0))
        screen.blit(epsilon_text, (10, 10 + 20 * 2)) # Adjusted Y coordinate
        episode_text = font.render(f"Episode: {episode + 1}/{total_episodes}", True, (0, 0, 0))
        screen.blit(episode_text, (10, 10 + 20 * 3)) # Adjusted Y coordinate
        learning_time = round(time.time() - start_time)
        minutes = learning_time // 60
        seconds = learning_time % 60
        time_text = font.render(f"Learning Time: {minutes:02d}:{seconds:02d}", True, (0, 0, 0))
        screen.blit(time_text, (10, 10 + 20 * 4)) # Adjusted Y coordinate
        average_moves_text = font.render(f"Average Moves: {average_moves:.2f}", True, (0, 0, 0))
        screen.blit(average_moves_text, (10, 10 + 20 * 5)) # Adjusted Y coordinate

        pygame.display.flip()
        clock.tick(10)  # Limit frame rate

    # Decay epsilon
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # Perform experience replay
    experience_replay()

    episodes_since_average += 1
    if episodes_since_average >= 10:
        average_moves = total_moves / 10  # Calculate average moves
        episodes_since_average = 0  # Reset count
        total_moves = 0  # Reset total moves
        print(f"Average amount of moves for last 10 steps: {average_moves}")

pygame.quit()
