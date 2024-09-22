import numpy as np
import random
import pickle
import time  # Import time module

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [' '] * 9  # Initialize a blank board
        self.current_winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board)

    def available_actions(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, position, player):
        if self.board[position] == ' ':
            self.board[position] = player
            if self.check_winner(position, player):
                self.current_winner = player
            return True
        return False

    def check_winner(self, position, player):
        # Check row
        row_index = position // 3
        row = self.board[row_index*3 : (row_index+1)*3]
        if all(spot == player for spot in row):
            return True
        # Check column
        col_index = position % 3
        column = [self.board[col_index + i*3] for i in range(3)]
        if all(spot == player for spot in column):
            return True
        # Check diagonals
        if position % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all(spot == player for spot in diagonal1):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all(spot == player for spot in diagonal2):
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def print_board(self):
        print("\nCurrent board:")
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')
        print()

class QLearningAgent:
    def __init__(self, player, epsilon=1.0, alpha=0.5, gamma=0.9):
        self.player = player  # 'X' or 'O'
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = {}  # State-action value table

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return random.choice(available_actions)
        else:
            # Exploit: choose the best known action
            q_values = [self.get_q_value(state, a) for a in available_actions]
            max_q = max(q_values)
            max_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
            return random.choice(max_actions)

    def update_q_value(self, state, action, reward, next_state, done):
        old_q = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            next_available_actions = [i for i, spot in enumerate(next_state) if spot == ' ']
            future_qs = [self.get_q_value(next_state, a) for a in next_available_actions]
            target = reward + self.gamma * max(future_qs, default=0)
        new_q = old_q + self.alpha * (target - old_q)
        self.q_table[(state, action)] = new_q

class MinimaxOpponent:
    def __init__(self, player):
        self.player = player  # 'X' or 'O'
        self.opponent = 'O' if player == 'X' else 'X'

    def choose_action(self, state, available_actions):
        # Convert state to list for mutability
        board = list(state)
        best_score = None
        best_move = None
        for action in available_actions:
            board[action] = self.player
            score = self.minimax(board, False)
            board[action] = ' '
            if best_score is None or score > best_score:
                best_score = score
                best_move = action
        return best_move

    def minimax(self, board, is_maximizing):
        winner = self.check_winner(board)
        if winner == self.player:
            return 1
        elif winner == self.opponent:
            return -1
        elif ' ' not in board:
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = self.player
                    score = self.minimax(board, False)
                    board[i] = ' '
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = self.opponent
                    score = self.minimax(board, True)
                    board[i] = ' '
                    best_score = min(score, best_score)
            return best_score

    def check_winner(self, board):
        # Check rows, columns, and diagonals for a winner
        for i in range(3):
            # Rows
            if board[i*3] == board[i*3+1] == board[i*3+2] != ' ':
                return board[i*3]
            # Columns
            if board[i] == board[i+3] == board[i+6] != ' ':
                return board[i]
        # Diagonals
        if board[0] == board[4] == board[8] != ' ':
            return board[0]
        if board[2] == board[4] == board[6] != ' ':
            return board[2]
        return None

def train(agent, opponent, episodes=20000):
    start_epsilon = agent.epsilon
    end_epsilon = 0.1
    decay_rate = (end_epsilon / start_epsilon) ** (1. / episodes)
    start_time = time.time()  # Record start time
    for episode in range(episodes):
        game = TicTacToe()
        state = game.get_state()
        done = False
        agent.epsilon *= decay_rate  # Decay epsilon
        while not done:
            # Agent's turn
            available_actions = game.available_actions()
            action = agent.choose_action(state, available_actions)
            game.make_move(action, agent.player)
            next_state = game.get_state()
            if game.current_winner == agent.player:
                agent.update_q_value(state, action, reward=1, next_state=next_state, done=True)
                break
            elif game.is_draw():
                agent.update_q_value(state, action, reward=0.5, next_state=next_state, done=True)
                break
            else:
                agent.update_q_value(state, action, reward=0, next_state=next_state, done=False)
            # Opponent's turn
            state = next_state
            available_actions = game.available_actions()
            if not available_actions:
                break
            opponent_action = opponent.choose_action(state, available_actions)
            game.make_move(opponent_action, opponent.player)
            next_state = game.get_state()
            if game.current_winner == opponent.player:
                agent.update_q_value(state, action, reward=-1, next_state=next_state, done=True)
                break
            elif game.is_draw():
                agent.update_q_value(state, action, reward=0.5, next_state=next_state, done=True)
                break
            else:
                pass
            state = next_state
        # Optional: Print progress
        if (episode + 1) % 1000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode +1}/{episodes}, Epsilon: {agent.epsilon:.4f}, Elapsed Time: {elapsed_time:.2f}s")

def play_against_agent(agent):
    game = TicTacToe()
    human_player = 'O' if agent.player == 'X' else 'X'
    game.print_board()
    while True:
        # Human's turn
        available_actions = game.available_actions()
        while True:
            try:
                human_action = int(input(f"Choose your move (0-8): "))
                if human_action in available_actions:
                    break
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Please enter a valid integer from 0 to 8.")
        game.make_move(human_action, human_player)
        game.print_board()
        if game.current_winner == human_player:
            print("You win!")
            break
        elif game.is_draw():
            print("It's a draw!")
            break
        # Agent's turn
        state = game.get_state()
        agent_action = agent.choose_action(state, game.available_actions())
        game.make_move(agent_action, agent.player)
        print(f"Agent chooses position {agent_action}")
        game.print_board()
        if game.current_winner == agent.player:
            print("Agent wins!")
            break
        elif game.is_draw():
            print("It's a draw!")
            break

if __name__ == '__main__':
    # Parameters
    episodes = 20000  # Increase training episodes
    alpha = 0.7       # Learning rate
    gamma = 0.9       # Discount factor

    # Initialize agents
    agent = QLearningAgent(player='X', epsilon=1.0, alpha=alpha, gamma=gamma)
    opponent = MinimaxOpponent(player='O')  # More sophisticated opponent

    # Train the agent
    print("Training the agent...")
    train(agent, opponent, episodes=episodes)
    print("Training completed.")

    # Save the trained model
    with open('tic_tac_toe_agent.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)
    print("Trained model saved as 'tic_tac_toe_agent.pkl'.")

    # Load the trained model (optional)
    # with open('tic_tac_toe_agent.pkl', 'rb') as f:
    #     agent.q_table = pickle.load(f)
    # print("Trained model loaded.")

    # Play against the agent
    print("Now you can play against the trained agent!")
    play_against_agent(agent)
