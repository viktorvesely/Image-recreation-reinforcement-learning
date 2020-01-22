import numpy as np
import tensorflow as tf

import model as md
import memory_units

class DQN:
    def __init__(self,
    num_states,
    num_actions,
    hidden_units,
    gamma,
    max_experiences,
    min_experiences,
    batch_size,
    num_agents
    ):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.RMSprop(0.001)
        self.gamma = gamma
        self.model = md.Model(num_actions, num_states, hidden_units)
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.min_experiences_per_unit = int(min_experiences / num_agents)
        self.max_experiences_per_unit = int(max_experiences / num_agents)
        self.memory_units = []
        for i in range(num_agents):
            self.memory_units.append(memory_units.Memory_unit(
                self,
                i,
                self.min_experiences_per_unit,
                self.max_experiences_per_unit
            ))
        

        self.model.compile(
            self.optimizer,
            loss = tf.keras.losses.MeanSquaredError()
        )

    def predict(self, inputs):
        reworked_inputs = np.atleast_2d(inputs.astype('float32'))
        return self.model(reworked_inputs)

    def train_on_memory_unit(self, TargetNet, experience):
        if len(experience['s']) < self.min_experiences_per_unit:
            return 0

        ids = np.random.randint(low=0, high=len(experience['s']), size=self.batch_size)
        states = np.asarray([experience['s'][i] for i in ids])
        actions = np.asarray([experience['a'][i] for i in ids])
        rewards = np.asarray([experience['r'][i] for i in ids])
        states_next = np.asarray([experience['s2'][i] for i in ids])
        dones = np.asarray([experience['done'][i] for i in ids])
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

    def train(self, TargetNet):
        for memory_unit in self.memory_units:
            self.train_on_memory_unit(TargetNet, memory_unit.experience)

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])
    

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
