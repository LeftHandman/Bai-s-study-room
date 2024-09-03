import random
import copy

class Env():
    def __init__(self):
        self.start = (
            ('b',),
            ('g', 'r'),
            (),
            ()
        )
        self.goal = (
            (),
            (),
            ('g','r', 'b'),
            ()
        )
        self.length = [1, 2, 3, 2]
        self.state = self.start

    def reset(self):
        self.state = self.start

    def is_goal(self):
        return self.state == self.goal

    def step(self, action):
        state_list = [list(stack) for stack in self.state]
        if action == '1_2' and state_list[0]:
            state_list[1].append(state_list[0].pop())
        elif action == '2_1' and state_list[1]:
            state_list[0].append(state_list[1].pop())
        elif action == '2_3' and state_list[1]:
            state_list[2].append(state_list[1].pop())
        elif action == '1_3' and state_list[0]:
            state_list[2].append(state_list[0].pop())
        elif action == '3_2' and state_list[2]:
            state_list[1].append(state_list[2].pop())
        elif action == '3_1' and state_list[2]:
            state_list[0].append(state_list[2].pop())

        self.state = tuple(tuple(stack) for stack in state_list)
        return self.state, 1 if self.is_goal() else 0, self.is_goal()

    def action_filter(self):
        _action = copy.deepcopy(self.action)
        state_list = [list(stack) for stack in self.state]
        if len(state_list[2]) == self.length[2]:
            return ['3_1', '3_2']
        if len(state_list[0]) == self.length[0]:
            if '3_1' in _action:
                _action.remove('3_1')
            if '2_1' in _action:
                _action.remove('2_1')
            if len(state_list[1]) == self.length[1]:
                if '1_2' in _action:
                    _action.remove('1_2')
                if '3_2' in _action:
                    _action.remove('3_2')
        elif len(state_list[1]) == self.length[1]:
            if '3_2' in _action:
                _action.remove('3_2')
            if '1_2' in _action:
                _action.remove('1_2')
            if len(state_list[0]) == 0:
                if '1_2' in _action:
                    _action.remove('1_2')
                if '1_3' in _action:
                    _action.remove('1_3')
        if len(state_list[0]) == 0:
            if '1_3' in _action:
                _action.remove('1_3')
            if '1_2' in _action:
                _action.remove('1_2')
        if len(state_list[1]) == 0:
            if '2_3' in _action:
                _action.remove('2_3')
            if '2_1' in _action:
                _action.remove('2_1')
        return _action


def q_learning(env, num_episode=10000, alpha=0.1, gamma=0.95,
               initial_epsilon=0.9, epsilon_decay=0.999, min_epsilon=0.1, max_step=10000):
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
                done = True

        # q值收敛时终止迭代
        if episode > num_episode * 0.1 and max_delta < 0.00001:
            # print(episode)
            break

        if episode > 0 and episode % 2 == 0:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)  # 确保 epsilon 不低于 0.1

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1} completed. Total reward: {total_reward}. Epsilon: {epsilon}, max_delt: {max_delta}")

    return Q, rewards, steps


def get_best_action_sequence(Q, env):
    env.reset()
    state = env.state
    best_action_sequence = []
    action_count = 0

    while not env.is_goal() and action_count < 100:
        if state in Q and Q[state]:
            action = max(Q[state], key=Q[state].get)
            best_action_sequence.append(action)
            state, _, _ = env.step(action)
            action_count += 1
        else:
            break

    return best_action_sequence, action_count



env = Env()
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

