import gym
import numpy as np
from RL_brain import DeepQNetwork
import time

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001, )

total_steps = 0

for i_episode in range(130):

    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]  # 取出元组中的第一个元素

    ep_r = 0
    while True:
        action = RL.choose_action(observation)

        observation_, reward, done, truncated, info = env.step(action)
        if isinstance(observation_, tuple):
            observation_ = observation_[0]  # 取出元组中的第一个元素

        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done or truncated:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

# 保存模型
RL.save_model("dqn_cartpole_model.ckpt")

RL.plot_cost()


# 演示代码
def demonstrate_model(env, RL):
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]

    while True:
        env.render()
        action = RL.choose_action(observation)
        observation, reward, done, truncated, info = env.step(action)
        if isinstance(observation, tuple):
            observation = observation[0]
        if done or truncated:
            break
        time.sleep(0.02)  # 加一个延迟以便更好地观察


# 加载模型并演示
RL.load_model("dqn_cartpole_model.ckpt")

# 设置贪婪策略演示
RL.epsilon = 1.0

# 将环境设置为人类可见模式
env = gym.make('CartPole-v1', render_mode="human")
env = env.unwrapped
demonstrate_model(env, RL)

env.close()  # 确保关闭环境
