import pickle
from tictactoe import TicTacToe, QLearningAgent, MinimaxOpponent, play_against_agent

def main():
    # Filepath for the trained model
    model_filepath = 'tic_tac_toe_agent.pkl'

    if not os.path.exists(model_filepath):
        print("No trained model found. Please train the agent first.")
        return

    # Load the trained model
    with open(model_filepath, 'rb') as f:
        q_table = pickle.load(f)
    print(f"Loaded trained model from '{model_filepath}'.")

    # Initialize the agent with the loaded Q-values
    agent = QLearningAgent(player='X', epsilon=0.1, alpha=0.7, gamma=0.9)
    agent.q_table = q_table

    # Play against the agent
    print("Now you can play against the trained agent!")
    play_against_agent(agent)

if __name__ == '__main__':
    main()
