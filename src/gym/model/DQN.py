import model as md
import numpy as np
import tensorflow as tf

class DQN:
    def __init__(self,
    num_states,
    num_actions,
    hidden_units,
    gamma,
    max_experiences,
    min_experiences,
    batch_size
    ):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.RMSprop(0.001)
        self.gamma = gamma
        self.model = md.Model(num_actions, num_states, hidden_units)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

        self.model.compile(
            self.optimizer,
            loss = tf.keras.losses.MeanSquaredError()
        )

    def observe(self, prev_state, action, reward, state, done):
        self.add_experience({
            's': prev_state,
            'a': action,
            'r': reward,
            's2': state,
            'done': done
            })
        self.train()
        

    def predict(self, inputs):
        reworked_inputs = np.atleast_2d(inputs.astype('float32'))
        return self.model(reworked_inputs)

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1
                )

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1
                )
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])
    
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
