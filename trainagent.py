import pickle
import os
import time
from tictactoe import TicTacToe, QLearningAgent, MinimaxOpponent, RandomOpponent, train

def main():
    # Filepath for the trained model
    model_filepath = 'tic_tac_toe_agent.pkl'
    
    # Parameters
    episodes = 50000  # Number of additional training episodes
    alpha = 0.7       # Learning rate
    gamma = 0.9       # Discount factor

    # Initialize the agent
    agent = QLearningAgent(player='X', epsilon=1.0, alpha=alpha, gamma=gamma)

    # Load existing Q-table if available
    if os.path.exists(model_filepath):
        with open(model_filepath, 'rb') as f:
            q_table = pickle.load(f)
        agent.q_table = q_table
        print(f"Loaded existing Q-table from '{model_filepath}'. Continuing training...")

    # Initialize opponents
    minimax_opponent = MinimaxOpponent(player='O')
    random_opponent = RandomOpponent(player='O')

    # Start training
    print(f"Training the agent for {episodes} additional episodes...")
    start_time = time.time()
    for episode in range(episodes):
        # Alternate opponents
        if episode % 2 == 0:
            opponent = minimax_opponent
        else:
            opponent = random_opponent
        train(agent, opponent, episodes=1)  # Train for 1 episode with current opponent
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    # Save the updated Q-table
    with open(model_filepath, 'wb') as f:
        pickle.dump(agent.q_table, f)
    print(f"Updated Q-table saved to '{model_filepath}'.")

if __name__ == '__main__':
    main()
