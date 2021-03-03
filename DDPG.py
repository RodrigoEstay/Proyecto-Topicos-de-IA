import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple
import pickle


class OUActionNoise:
	def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
		self.theta = theta
		self.mean = mean
		self.std_dev = std_deviation
		self.dt = dt
		self.x_initial = x_initial
		self.reset()

	def __call__(self):
		# Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
		x = (self.x_prev
			+ self.theta * (self.mean - self.x_prev) * self.dt
			+ self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
		# Store x into x_prev
		# Makes next noise dependent on current one
		self.x_prev = x
		return x

	def reset(self):
		if self.x_initial is not None:
			self.x_prev = self.x_initial
		else:
			self.x_prev = np.zeros_like(self.mean)


class Buffer:
	def __init__(self, buffer_capacity=100000, batch_size=64):
		# Number of "experiences" to store at max
		self.buffer_capacity = buffer_capacity
		# Num of tuples to train on.
		self.batch_size = batch_size

		# Its tells us num of times record() was called.
		self.buffer_counter = 0

		# Instead of list of tuples as the exp.replay concept go
		# We use different np.arrays for each tuple element
		self.state_buffer = np.zeros((self.buffer_capacity, 28))
		self.action_buffer = np.zeros((self.buffer_capacity, 4))
		self.reward_buffer = np.zeros((self.buffer_capacity, 1))
		self.next_state_buffer = np.zeros((self.buffer_capacity, 28))

	# Takes (s,a,r,s') obervation tuple as input
	def record(self, obs_tuple):
		# Set index to zero if buffer_capacity is exceeded,
		# replacing old records
		index = self.buffer_counter % self.buffer_capacity

		self.state_buffer[index] = obs_tuple[0]
		self.action_buffer[index] = obs_tuple[1]
		self.reward_buffer[index] = obs_tuple[2]
		self.next_state_buffer[index] = obs_tuple[3]

		self.buffer_counter += 1

	# Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
	# TensorFlow to build a static graph out of the logic and computations in our function.
	# This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
	@tf.function
	def update(self, state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model, target_actor, target_critic, critic_optimizer, actor_optimizer,):
		# Training and updating Actor & Critic networks.
		# See Pseudo Code.
		with tf.GradientTape() as tape:
			target_actions = target_actor(next_state_batch, training=True)
			y = reward_batch + 0.98 * target_critic(tf.concat([next_state_batch, tf.cast(target_actions, dtype=tf.float64)], 1), training=True)
			#y = reward_batch + 0.98 * target_critic([next_state_batch, target_actions], training=True)
			critic_value = critic_model(tf.concat([state_batch, tf.cast(action_batch, dtype=tf.float64)], 1), training=True)
			#critic_value = critic_model([state_batch, action_batch], training=True)
			critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

		critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
		critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

		with tf.GradientTape() as tape:
			actions = actor_model(state_batch, training=True)
			critic_value = critic_model(tf.concat([state_batch, tf.cast(actions, dtype=tf.float64)], 1), training=True)
			#critic_value = critic_model([state_batch, actions], training=True)
			# Used `-value` as we want to maximize the value given
			# by the critic for our actions
			actor_loss = -tf.math.reduce_mean(critic_value)

		actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
		actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

	# We compute the loss and update parameters
	def learn(self, actor_model, critic_model, target_actor, target_critic, critic_optimizer, actor_optimizer):
		# Get sampling range
		record_range = min(self.buffer_counter, self.buffer_capacity)
		# Randomly sample indices
		batch_indices = np.random.choice(record_range, self.batch_size)

		# Convert to tensors
		state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
		action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
		reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

		self.update(state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model, target_actor, target_critic, critic_optimizer, actor_optimizer)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
	for (a, b) in zip(target_weights, weights):
		a.assign(b * tau + a * (1 - tau))

def get_actor():
	# Initialize weights between -3e-3 and 3-e3
	last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

	inputs = Input(shape=(28,))
	out = Dense(256, activation="relu")(inputs)
	out = Dense(256, activation="relu")(out)
	out = Dense(256, activation="relu")(out)
	outputs = Dense(4, activation="tanh", kernel_initializer=last_init)(out)

	model = tf.keras.Model(inputs, outputs)
	return model


def get_critic():

	inputs = Input(shape=(28 + 4,))
	out = Dense(256, activation="relu")(inputs)
	out = Dense(256, activation="relu")(out)
	out = Dense(256, activation="relu")(out)
	outputs = Dense(1, activation="linear")(out)

	model = tf.keras.Model(inputs, outputs)
	# State as input
	'''
	state_input = Input(shape=(28))
	state_out = Dense(32, activation="relu")(state_input)
	state_out = Dense(64, activation="relu")(state_out)

	# Action as input
	action_input = Input(shape=(4))
	action_out = Dense(64, activation="relu")(action_input)

	# Both are passed through seperate layer before concatenating
	concat = tf.keras.layers.Concatenate()([state_out, action_out])

	out = Dense(256, activation="relu")(concat)
	out = Dense(256, activation="relu")(out)
	outputs = Dense(1, activation="linear")(out)

	model = tf.keras.Model([state_input, action_input], outputs)
	'''

	return model


class DDPG:

	def __init__(self):
		std_dev = 0.05
		self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

		self.actor_model = get_actor()
		self.critic_model = get_critic()

		self.target_actor = get_actor()
		self.target_critic = get_critic()

		# Making the weights equal initially
		self.target_actor.set_weights(self.actor_model.get_weights())
		self.target_critic.set_weights(self.critic_model.get_weights())

		# Learning rate for actor-critic models
		critic_lr = 0.001
		actor_lr = 0.001

		self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
		self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

		# Discount factor for future rewards
		self.gamma = 0.98
		# Used to update target networks
		self.tau = 0.05

		self.buffer = Buffer(50000, 64)

		self.optLoad = False


	def policy(self, state):
		sampled_actions = tf.squeeze(self.actor_model(state))
		noise = self.ou_noise()
		# Adding noise to action
		sampled_actions = sampled_actions.numpy() + noise

		# We make sure action is within bounds
		legal_action = np.clip(sampled_actions, -1.0, 1.0)

		return np.squeeze(legal_action)


	def act(self, obs):
		obs = np.squeeze(obs)
		tf_obs = tf.expand_dims(tf.convert_to_tensor(obs), 0)
		action = self.policy(tf_obs)
		return action

	def record(self, prevObs, action, reward, obs):
		self.buffer.record((prevObs, action, reward, obs))

	def train(self):
		self.buffer.learn(self.actor_model, self.critic_model, self.target_actor, self.target_critic, self.critic_optimizer, self.actor_optimizer)

		update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
		update_target(self.target_critic.variables, self.critic_model.variables, self.tau)

		if self.optLoad:
			self.loadOpts()

	def save(self, dir):
		self.actor_model.save_weights(dir + "_actor_model")
		self.critic_model.save_weights(dir + "_critic_model")
		self.target_actor.save_weights(dir + "_target_actor")
		self.target_critic.save_weights(dir + "_target_critic")

		with open(dir + "_optimizers.pickle", "wb") as f:
			pickle.dump([self.actor_optimizer.get_weights(), self.critic_optimizer.get_weights()], f)

	def load(self, dir):
		try:
			self.actor_model.load_weights(dir + "_actor_model")
		except:
			print("ERROR AL CARGAR ACTOR MODEL")
		try:
			self.critic_model.load_weights(dir + "_critic_model")
		except:
			print("ERROR AL CARGAR CRITIC MODEL")
		try:
			self.target_actor.load_weights(dir + "_target_actor")
		except:
			print("ERROR AL CARGAR TARGET ACTOR")
		try:
			self.target_critic.load_weights(dir + "_target_critic")
		except:
			print("ERROR AL CARGAR TARGET CRITIC")
		self.optLoad = True
		self.optDir = dir + "_optimizers.pickle"

	def loadOpts(self):
		try:
			with open(self.optDir, "rb") as f:
				actorOptWeights, criticOptWeights= pickle.load(f)
			self.actor_optimizer.set_weights(actorOptWeights)
			self.critic_optimizer.set_weights(criticOptWeights)
			print("CARGADOS LOS OPTIMIZER!")
		except:
			print("ERROR AL CARGAR LOS OPTIMIZER")
		self.optLoad = False
