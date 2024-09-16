import pickle
import os
import time
from tictactoe import TicTacToe, QLearningAgent, MinimaxOpponent, train

def main():
    # Filepath for the trained model
    model_filepath = 'tic_tac_toe_agent.pkl'
    
    # Parameters
    episodes = 50000  # Number of training episodes
    alpha = 0.7       # Learning rate
    gamma = 0.9       # Discount factor

    # Initialize the agent and opponent
    agent = QLearningAgent(player='X', epsilon=1.0, alpha=alpha, gamma=gamma)
    opponent = MinimaxOpponent(player='O')

    print("Training the agent...")
    start_time = time.time()
    train(agent, opponent, episodes=episodes)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    # Save the trained model
    with open(model_filepath, 'wb') as f:
        pickle.dump(agent.q_table, f)
    print(f"Trained model saved as '{model_filepath}'.")

if __name__ == '__main__':
    main()
