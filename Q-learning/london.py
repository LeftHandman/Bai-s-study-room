import pygame
import random
import copy
# 初始化 Pygame
pygame.init()

# 设置窗口大小
window_size = (600, 400)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Q-learning Optimal Path")

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (139, 0, 139)
# 定义方块大小和位置
block_size = 50
positions = [(100, 100), (200, 100), (300, 100),
             (400, 100), (500, 100)]

class Env():
    def __init__(self, plate=None):
        if plate is None:
            plate = ['r', 'g', 'b', 'y', 'p']
        self.start = []
        self.plate = copy.copy(plate)
        self.length = [ 2, 3, 2, 1]
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


def q_learning(env, num_episode=10000, alpha=0.1, gamma=0.95,
               initial_epsilon=0.9, epsilon_decay=0.999, min_epsilon=0.1, max_step=100000):
    Q = {}
    epsilon = initial_epsilon
    rewards = []
    steps = []

    for episode in range(num_episode):
        env.reset()
        step_count = 0
        done = False
        max_delta = 0
        total_reward = 0
        while not done:
            state = env.state
            if state not in Q:
                Q[state] = {}
            if not Q[state]:  # 如果Q[state]为空
                action = random.choice(env.action_filter())
                Q[state][action] = 0
            else:
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(env.action_filter())
                    if action not in Q[state]:
                        Q[state][action] = 0
                else:
                    action = max(Q[state], key=Q[state].get)  # 利用

            next_state, reward, done = env.step(action)
            if next_state not in Q:
                Q[next_state] = {}
            if Q[next_state]:
                best_next_action = max(Q[next_state], key=Q[next_state].get)
            else:
                best_next_action = None
            td_target = reward + gamma * Q[next_state].get(best_next_action, 0)
            td_error = td_target - Q[state].get(action, 0)
            Q[state][action] += alpha * td_error

            max_delta = max(max_delta, abs(td_error))
            total_reward += reward
            step_count += 1

            rewards.append(total_reward)
            steps.append(step_count)
            if step_count >= max_step:
                return Q, 0, -1

        # q值收敛时终止迭代
        if episode > num_episode * 0.5 and max_delta < 0.00001:
            print(episode)
            break

        if episode > 0 and episode % 5 == 0:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)  # 确保 epsilon 不低于 0.1

        if (episode + 1) % 1000 == 0:
            print(
                f"Episode {episode + 1} completed. Total reward: {total_reward}. Epsilon: {epsilon}, max_delt: {max_delta}")

    return Q, rewards, steps


def get_best_action_sequence(Q, env):
    env.reset()
    state = env.state
    best_action_sequence = []
    action_count = 0

    while not env.is_goal() and action_count < 1000:
        if state in Q and Q[state]:
            action = max(Q[state], key=Q[state].get)
            best_action_sequence.append(action)
            state, _, _ = env.step(action)
            action_count += 1
        else:
            break

    return best_action_sequence, action_count

def draw_state(state):
    screen.fill(WHITE)
    for i, stack in enumerate(state):
        for j, block in enumerate(stack):
            color = BLACK
            if block == 'b':
                color = BLUE
            elif block == 'g':
                color = GREEN
            elif block == 'r':
                color = RED
            elif block == 'y':
                color = YELLOW
            elif block == 'p':
                color = PURPLE
            pygame.draw.rect(screen, color, pygame.Rect(positions[i][0], positions[i][1] - j * block_size, block_size, block_size))
    pygame.display.flip()

env = Env()
env.start = ((),('r','g','y',), ('p','b'), ())
env.goal = ((), ('p','b','y'), ('r', 'g'), ())
#
env.state = copy.copy(env.start)
Q, _, _ = q_learning(env)
best_action_sequence, action_count = get_best_action_sequence(Q, env)

print("Q-table:")
for state, actions in Q.items():
    print(f"State: {state}")
    for action, value in actions.items():
        print(f"  Action: {action}, Value: {value}")

print("\nStart State:", env.start)
print("Goal State:", env.goal)
print("\nBest Action Sequence:", best_action_sequence)
print("Number of Actions:", action_count)

# 渲染最优路径
running = True
current_step = 0
env.reset()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if current_step < len(best_action_sequence):
        draw_state(env.state)
        pygame.time.wait(1000)  # 等待500毫秒
        env.step(best_action_sequence[current_step])
        current_step += 1
    else:
        draw_state(env.state)
        pygame.time.wait(2000)  # 等待2秒
        running = False

pygame.quit()
