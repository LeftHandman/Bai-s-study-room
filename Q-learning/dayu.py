import copy
import random


# 迷宫环境定义
class MazeEnv:
    def __init__(self, difficulty):
        self.level = {1: {'height': 4, 'width': 4, 'mountain': 0, 'swamp': 0},
                      2: {'height': 5, 'width': 5, 'mountain': 3, 'swamp': 0},
                      3: {'height': 5, 'width': 5, 'mountain': 0, 'swamp': 3},
                      4: {'height': 5, 'width': 5, 'mountain': 7, 'swamp': 7},
                      5: {'height': 6, 'width': 6, 'mountain': 3, 'swamp': 4},
                      6: {'height': 7, 'width': 6, 'mountain': 4, 'swamp': 5},
                      7: {'height': 8, 'width': 6, 'mountain': 5, 'swamp': 6},
                      8: {'height': 9, 'width': 6, 'mountain': 10, 'swamp': 12}}
        self.level = self.level[difficulty]
        self.maze = []
        for j in range(self.level['height']):
            row = ['.'] * self.level['width']  # 创建新的行列表
            self.maze.append(row)

        self.start, self.goal = self._generate_start_goal()
        self.maze[self.start[0]][self.start[1]] = 'S'
        self.maze[self.goal[0]][self.goal[1]] = 'G'

        for m in range(self.level['mountain']):
            success = False
            while not success:
                row = random.randint(0, self.level['height'] - 1)
                col = random.randint(0, self.level['width'] - 1)
                if self.maze[row][col] == '.':
                    self.maze[row][col] = '#'
                    success = True

        for s in range(self.level['swamp']):
            success = False
            while not success:
                row = random.randint(0, self.level['height'] - 1)
                col = random.randint(0, self.level['width'] - 1)
                if self.maze[row][col] == '.':
                    self.maze[row][col] = '~'
                    success = True

        self.state = self.start
        self.actions = ['up', 'down', 'left', 'right', 'x_up', 'x_down', 'x_left', 'x_right']

    def _generate_start_goal(self):
        def random_edge_position():
            edge = random.choice(['top', 'bottom', 'left', 'right'])
            if edge == 'top':
                return (0, random.randint(0, self.level['width'] - 1))
            elif edge == 'bottom':
                return (self.level['height'] - 1, random.randint(0, self.level['width'] - 1))
            elif edge == 'left':
                return (random.randint(0, self.level['height'] - 1), 0)
            elif edge == 'right':
                return (random.randint(0, self.level['height'] - 1), self.level['width'] - 1)

        start = random_edge_position()
        goal = random_edge_position()
        while goal == start:  # 确保起点和终点不相同
            goal = random_edge_position()
        return start, goal

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        row, col = self.state
        if action == 'up':
            row -= 1
        elif action == 'down':
            row += 1
        elif action == 'left':
            col -= 1
        elif action == 'right':
            col += 1
        elif action == 'x_up':
            while row > 0 and self.maze[row - 1][col] != '#':
                row -= 1
                if row == 0:
                    break
                elif self.maze[row][col] == 'G':
                    break
                elif self.maze[row - 1][col] == '#':
                    break
            else:
                return self.state, 0, False
        elif action == 'x_down':
            while row < len(self.maze) - 1 and self.maze[row + 1][col] != '#':
                row += 1
                if row == self.level['height'] - 1:
                    break
                elif self.maze[row + 1][col] == '#':
                    break
                elif self.maze[row][col] == 'G':
                    break
            else:
                return self.state, 0, False
        elif action == 'x_left':
            while col > 0 and self.maze[row][col - 1] != '#':
                col -= 1
                if col == 0:
                    break
                elif self.maze[row][col - 1] == '#':
                    break
                elif self.maze[row][col] == 'G':
                    break
            else:
                return self.state, 0, False
        elif action == 'x_right':
            while col < len(self.maze[0]) - 1 and self.maze[row][col + 1] != '#':
                col += 1
                if col == self.level['width'] - 1:
                    break
                elif self.maze[row][col + 1] == '#':
                    break
                elif self.maze[row][col] == 'G':
                    break
            else:
                return self.state, 0, False

        # 检查是否超出边界或撞墙
        if row < 0 or row >= len(self.maze) or col < 0 or col >= len(self.maze[0]) or self.maze[row][col] == '#':
            return self.state, 0, False  # 保持在原地，负奖励
        else:
            self.state = (row, col)
            if self.maze[row][col] == '~':
                if action in ['x_up', 'x_down', 'x_left', 'x_right']:
                    return self.state, 0, False  # 踩到沼泽，等待一步，奖励为0
                else:
                    return self.state, 0, False  # 踩到沼泽，等待一步，奖励为0
            elif self.state == self.goal:
                return self.state, 1, True  # 到达目标，奖励为1
            else:
                return self.state, 0, False  # 非沼泽普通位置，轻微负奖励

    def is_goal(self, state):
        return state == self.goal


# Q-learning 算法实现
def q_learning(env, num_episodes, alpha, gamma, initial_epsilon, epsilon_decay, initial_max_steps):
    # 初始化 Q 表
    Q = {}
    for row in range(len(env.maze)):
        for col in range(len(env.maze[0])):
            _actions = copy.deepcopy(env.actions)

            if row == 0 or env.maze[row-1][col] == '#':
                _actions.remove('up')
                _actions.remove('x_up')
            if col == 0 or env.maze[row][col-1] == '#':
                _actions.remove('left')
                _actions.remove('x_left')
            if row == len(env.maze) - 1 or env.maze[row+1][col] == '#':
                _actions.remove('down')
                _actions.remove('x_down')
            if col == len(env.maze[0]) - 1 or env.maze[row][col+1] == '#':
                _actions.remove('right')
                _actions.remove('x_right')
            if env.maze[row][col] not in ['#']:
                for a in env.actions:
                    _a = copy.deepcopy(env.actions)
                    if a == 'up' or a == 'x_up':
                        _a.remove('down')
                        _a.remove('x_down')
                    elif a == 'down' or a == 'x_down':
                        _a.remove('up')
                        _a.remove('x_up')
                    elif a == 'left' or a == 'x_left':
                        _a.remove('right')
                        _a.remove('x_right')
                    elif a == 'right' or a == 'x_right':
                        _a.remove('left')
                        _a.remove('x_left')
                    inter = list(set(_a).intersection(set(_actions)))
                    Q[((row, col), a)] = {x: 0 for x in inter}
            if env.maze[row][col] == 'S':
                Q[((row, col), None)] = {x: 0 for x in _actions}
    epsilon = initial_epsilon
    rewards = []
    steps = []
    max_steps = initial_max_steps

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_delta = 0  # 用于追踪 Q 值的最大变化
        last_action = None
        while not done:
            if not list(Q[state, last_action].keys()):
                break
            # ε-贪婪策略选择动作
            if random.uniform(0, 1) < epsilon:
                action = random.choice(list(Q[state, last_action].keys()))
            else:

                action = max(Q[(state, last_action)], key=Q[(state, last_action)].get)  # 利用
            next_state, reward, done = env.step(action)

            # 更新 Q 值
            if not list(Q[next_state, action].keys()):
                break
            best_next_action = max(Q[(next_state, action)], key=Q[(next_state, action)].get)
            if env.maze[next_state[0]][next_state[1]] == '~' or action in ['x_up', 'x_down', 'x_left', 'x_right']:
                td_target = (reward + gamma * Q[(next_state, action)][best_next_action]) * gamma
            else:
                td_target = reward + gamma * Q[(next_state, action)][best_next_action]
            td_error = td_target - Q[(state, last_action)][action]
            Q[(state, last_action)][action] += alpha * td_error

            # 追踪最大 Q 值变化
            max_delta = max(max_delta, abs(td_error))
            last_action = action

            state = next_state
            total_reward += reward
            step_count += 1

            # 如果超过最大步数上限，认为地图无解，终止当前episode
            if step_count >= max_steps:
                return Q, 0, 0, 0
        rewards.append(total_reward)
        steps.append(step_count)

        # q值收敛时终止迭代
        if episode > num_episodes * 0.5 and max_delta < 0.00001:
            break
        # ε 衰减
        if episode > 0 and episode % 12 == 0:
            epsilon = max(0.01, epsilon * epsilon_decay)  # 确保 epsilon 不低于 0.1
        # 更新最大步数上限，随 epsilon 衰减
        max_steps = max(1, int(initial_max_steps * epsilon))
        # 打印进度
        if (episode + 1) % 100000 == 0:
            print(
                f"Episode {episode + 1} completed. Total reward: {total_reward}. Epsilon: {epsilon}, max_delt: {max_delta}")

    optimal_steps = compute_optimal_steps(env, Q)
    return Q, rewards, steps, optimal_steps


def compute_optimal_steps(env, Q):
    state = env.reset()  # 确保环境状态重置到起点
    steps = 0
    last_action = None
    while state != env.goal:
        print(state)
        action = max(Q[(state, last_action)], key=Q[(state, last_action)].get)
        print(action)
        state, reward, _ = env.step(action)
        steps += 2 if env.maze[state[0]][state[1]] == '~' or action in ['x_up', 'x_down', 'x_left', 'x_right'] else 1
        last_action = action
    return steps


# 设置参数并运行 Q-learning
env = MazeEnv(8)
env.maze = [
['G', '~', '~', 'S', '.', '~'] ,
['.', '~', '.', '#', '.', '.'] ,
['.', '#', '.', '.', '.', '.'] ,
['#', '#', '.', '~', '~', '.'] ,
['.', '.', '.', '#', '#', '.'] ,
['~', '~', '.', '.', '.', '#'] ,
['.', '~', '~', '~', '~', '.'] ,
['#', '.', '#', '.', '.', '#'] ,
['.', '.', '.', '.', '.', '.'] , ]
env.start = (0, 3)
env.goal = (0, 0)
num_episodes = 50000  # 增加最大迭代次数
alpha = 0.1
gamma = 0.9
initial_epsilon = 0.8  # 初始ε
epsilon_decay = 0.9999  # 衰减率
initial_max_steps = 100000  # 设置最大步数上限

for row in env.maze:
    print(row, ", ")

Q, rewards, steps, optimal_steps = q_learning(env, num_episodes, alpha, gamma, initial_epsilon, epsilon_decay,
                                              initial_max_steps)

# 打印从起点到目标的最优步数
print(f"Optimal steps from start to goal: {optimal_steps}")
