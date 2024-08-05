import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

# Define the environment (chatbot) class
class ChatbotEnvironment:
    def __init__(self):
        self.actions = ['Hello', 'How are you?', 'Goodbye']
        self.state_space_size = len(self.actions)
        self.current_state = 0

    def reset(self):
        self.current_state = 0

    def step(self, action_index):
        # Define the transition logic here (e.g., define how the chatbot responds)
        if action_index == 0:
            response = "Hello! How can I assist you?"
        elif action_index == 1:
            response = "I'm doing well, thank you for asking."
        else:
            response = "Goodbye! Have a great day."
        
        # Calculate reward (you can customize this based on your task)
        reward = 1 if action_index == 2 else 0

        # Update the state (for this simple example, the state is the same as the action index)
        self.current_state = action_index

        return response, reward

# Define the Deep Q-Learning agent
class DQLAgent:
    def __init__(self, state_space_size, action_space_size):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_space_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_space_size, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space_size)
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.array(random.sample(self.memory, batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training the DQL agent
env = ChatbotEnvironment()
state_space_size = env.state_space_size
action_space_size = len(env.actions)
agent = DQLAgent(state_space_size, action_space_size)
batch_size = 32
episodes = 1000

for episode in range(episodes):
    state = np.reshape(env.current_state, [1, state_space_size])
    for t in range(100):
        action = agent.act(state)
        response, reward = env.step(action)
        next_state = np.reshape(env.current_state, [1, state_space_size])
        agent.remember(state, action, reward, next_state, False)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if reward == 1:
            print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))
            break

# You can now interact with the chatbot after training
while True:
    user_input = input("You: ")
    state = np.reshape(env.current_state, [1, state_space_size])
    action = agent.act(state)
    response, _ = env.step(action)
    print("Chatbot:", response)
