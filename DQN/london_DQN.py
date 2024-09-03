import random
import numpy as np
import copy
import tensorflow as tf
from collections import deque
from tensorflow import keras

from keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Env():
    def __init__(self, plate=None):
        if plate is None:
            plate = ['r', 'g', 'b', 'y', 'p']
        self.start = []
        self.plate = copy.copy(plate)
        self.length = [2, 3, 2, 1]
        for l in range(len(self.length)):
            self.start.append([])
        self.goal = copy.deepcopy(self.start)
        for i in range(len(self.plate)):
            success = False
            while not success:
                rand_num = random.randint(0, len(self.length) - 1)
                if len(self.start[rand_num]) < self.length[rand_num]:
                    if 0 < len(self.plate) - i:
                        self.start[rand_num].append(self.plate.pop(random.randint(0, len(self.plate)-i-1)))
                    else:
                        self.start[rand_num].append(self.plate.pop())
                    success = True
        self.plate = copy.copy(plate)
        for j in range(len(self.plate)):
            success = False
            while not success:
                rand_num = random.randint(0, len(self.length) - 1)
                if len(self.goal[rand_num]) < self.length[rand_num]:
                    if 0 < len(self.plate) - j:
                        self.goal[rand_num].append(self.plate.pop(random.randint(0, len(self.plate) - j - 1)))
                    else:
                        self.goal[rand_num].append(self.plate.pop())
                    success = True
        self.goal = tuple(tuple(stack) for stack in self.goal)
        self.start = tuple(tuple(stack) for stack in self.start)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def is_goal(self):
        return self.state == self.goal

    def step(self, action):
        state_list = [list(stack) for stack in self.state]
        state_list[action[1]-1].append(state_list[action[0]-1].pop())
        self.state = tuple(tuple(stack) for stack in state_list)
        return self.state, 1 if self.is_goal() else 0, self.is_goal()

    def action_filter(self):
        _action = []
        state_list = [list(stack) for stack in self.state]
        for i in range(len(state_list)):
            if self.length[i] > len(state_list[i]):
                for j in range(len(state_list)):
                    if len(state_list[j]) and i != j:
                        _action.append(tuple((int(j)+1, int(i)+1)))
        return _action

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣率
        self.epsilon = 1.0  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # 神经网络用于近似 Q 函数
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # 复制模型权重到目标网络
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # 返回 Q 值最大的动作

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def train_dqn(env, episodes=1000, batch_size=32):
    state_size = len(env.start) * max(env.length)
    action_size = len(env.action_filter())
    agent = DQNAgent(state_size, action_size)
    done = False

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(f"Episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save(f"dqn_model_{e}.h5")

env = Env()
train_dqn(env)
