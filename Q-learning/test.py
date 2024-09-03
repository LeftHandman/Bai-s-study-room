from new_dayu import q_learning, MazeEnv
import json

# 地图翻译对照字典
translation = {
    '.': 0,
    'S': 1,
    '~': 2,
    '#': 3,
    'G': 4
}

max_steps = 0
for i in range(30000):
    env = MazeEnv(7)
    num_episodes = 60000  # 增加最大迭代次数
    alpha = 0.1
    gamma = 0.9
    initial_epsilon = 1.0
    epsilon_decay = 0.999  # 衰减率
    initial_max_steps = 500000  # 设置最大步数上限

    Q, rewards, steps, max_steps = q_learning(env, num_episodes, alpha, gamma, initial_epsilon, epsilon_decay,
                                                  initial_max_steps)
    if max_steps >= 17:
        print("----------------------------")
        translated_map = [[translation[cell] for cell in row] for row in env.maze]

        # 打印翻译后的地图
        print('[')
        for row in translated_map:
            print('    [', end='')
            for i, cell in enumerate(row):
                if i < len(row) - 1:
                    print(cell, end=', ')
                else:
                    print(cell, end='')
            print('],')
        print(']')

        # 打印从起点到目标的最优步数
        print(f"Optimal steps from start to goal: {max_steps}")


