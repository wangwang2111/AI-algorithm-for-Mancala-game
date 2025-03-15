import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from ai.rules import initialize_board, get_valid_moves, make_move, is_terminal

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DQNAgent:
    def __init__(self, state_size=14, action_size=6):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.002
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Neural Network with proper input shape"""
        model = Sequential([
            Dense(32, input_shape=(self.state_size,), activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def _preprocess_state(self, board):
        """Add batch dimension and normalize"""
        return np.array(board).reshape(1, -1) / 4.0

    def get_action(self, state, valid_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        state = self._preprocess_state(state)
        act_values = self.model.predict(state, verbose=0)
        return valid_moves[np.argmax(act_values[0][valid_moves])]

    def replay(self, batch_size=128):
        minibatch = random.sample(self.memory, batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            state = self._preprocess_state(state)
            next_state = self._preprocess_state(next_state)
            
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            states.append(state[0])
            targets.append(target[0])
        
        # Batch training
        self.model.fit(
            np.array(states), 
            np.array(targets), 
            epochs=1, 
            verbose=0
        )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=1000, batch_size=128):
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                valid_moves = env.get_valid_moves()
                action = self.get_action(state, valid_moves)
                next_state, reward, done = env.step(action)
                
                # Store original state without batch dimension
                self.remember(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {self.epsilon:.2f}")
                    if len(self.memory) > batch_size:
                        self.replay(batch_size)
                    self.update_target_model()
                    break

class MancalaEnv:
    def __init__(self):
        self.board = initialize_board()
        self.current_player = "player_1"
    
    def reset(self):
        self.board = initialize_board()
        return self._get_state()
    
    def _get_state(self):
        """Proper state representation"""
        return np.array(
            self.board["player_1"][:6] + 
            [self.board["player_1"][6]] + 
            self.board["player_2"][:6] + 
            [self.board["player_2"][6]]
        )
    
    def get_valid_moves(self):
        return get_valid_moves(self.board, self.current_player)
    
    def step(self, action):
        """Execute one step in the environment"""
        new_board, extra_turn = make_move(self.board, self.current_player, action)
        self.board = new_board
        reward = self._calculate_reward()
        done = is_terminal(self.board)
        
        if not extra_turn:
            self.current_player = "player_2" if self.current_player == "player_1" else "player_1"
        
        return self._get_state(), reward, done
    
    def _calculate_reward(self):
        """Custom reward function"""
        score_diff = self.board["player_2"][6] - self.board["player_1"][6]
        
        # Intermediate rewards
        reward = score_diff * 0.1  # Small reward for leading
        
        if is_terminal(self.board):
            if self.board["player_2"][6] > 24:
                return 1  # Win
            elif self.board["player_1"][6] > 24:
                return -1  # Lose
            else:
                return 0.5 if score_diff > 0 else -0.5  # Draw with advantage
        
        return reward

# Usage
if __name__ == "__main__":
    env = MancalaEnv()
    agent = DQNAgent()
    
    # Train the agent
    agent.train(env, episodes=1000)
    
    # Save the trained model
    agent.model.save("mancala_dqn.h5")