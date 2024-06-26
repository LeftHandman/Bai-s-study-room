import numpy as np
import tensorflow.compat.v1 as tf

# 禁用TensorFlow 2.x默认行为
tf.disable_v2_behavior()

class DeepQNetwork:
    def __init__(
            self,
            n_actions,  # 动作空间大小
            n_features,  # 状态特征数量
            learning_rate=0.01,  # 学习率
            reward_decay=0.9,  # 奖励折扣因子
            e_greedy=0.9,  # ε-greedy策略的初始ε
            replace_target_iter=300,  # 替换目标网络的迭代次数
            memory_size=500,  # 经验回放记忆库大小
            batch_size=32,  # 批处理大小
            e_greedy_increment=None,  # ε-greedy策略的增量
            output_graph=False,  # 是否输出TensorBoard图
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 初始化记忆库 [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # 记录学习步数 (用于判断是否更新目标网络)
        self.learn_step_counter = 0

        # 创建强化学习网络
        self._build_net()

        # 替换目标网络的参数
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # 初始化TensorFlow会话
        self.sess = tf.Session()

        if output_graph:
            # 输出TensorBoard文件
            tf.summary.FileWriter("logs/", self.sess.graph)

        # 初始化所有TensorFlow变量
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # ------------------ 创建评估网络 ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 输入状态
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 目标Q值

        with tf.variable_scope('eval_net'):
            # 评估网络，简单的全连接层
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # 第一层全连接层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # 第二层全连接层，输出动作空间的Q值
            with tf.variable_scope('l2'):
                self.q_eval = tf.layers.dense(l1, self.n_actions, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q')

        with tf.variable_scope('loss'):
            # 损失函数，使用均方误差
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            # 优化器，使用Adam优化器
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ 创建目标网络 ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # 下一个状态

        with tf.variable_scope('target_net'):
            # 目标网络，与评估网络结构相同
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # 第一层全连接层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # 第二层全连接层，输出动作空间的Q值
            with tf.variable_scope('l2'):
                self.q_next = tf.layers.dense(l1, self.n_actions, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q')

    def store_transition(self, s, a, r, s_):
        # 存储经验 [s, a, r, s_]
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # 选择动作
        observation = observation[np.newaxis, :]  # 给状态添加一个新维度
        if np.random.uniform() < self.epsilon:
            # 随机选择动作
            action = np.random.randint(0, self.n_actions)
        else:
            # 根据评估网络选择最优动作
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        return action

    def learn(self):
        # 学习，更新评估网络的参数
        # 替换目标网络的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\n目标网络参数已更新\n')

        # 从记忆库中随机抽取批量数据
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 计算目标Q值
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # 下一个状态
                self.s: batch_memory[:, :self.n_features]  # 当前状态
            })

        # 更新Q目标
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 训练评估网络
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        # 可选：绘制损失曲线
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.ylabel('Cost')
        plt.xlabel('Training steps')
        plt.show()

    def save_model(self, model_path):
        # 保存模型
        saver = tf.train.Saver()
        saver.save(self.sess, model_path)
        print(f"模型已保存至 {model_path}")

    def load_model(self, model_path):
        # 加载模型
        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        print(f"模型已从 {model_path} 加载")

    def close(self):
        # 关闭TensorFlow会话
        self.sess.close()
