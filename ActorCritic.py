import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
import tensorflow_probability as tfp
from typing import Any, List, Sequence, Tuple
import pickle


class Actor(tf.keras.Model):

	# Utilizamos tensorflow.keras.layers.Dense para crear las redes.
	def __init__(self):
		super().__init__()
		self.l1 = Dense(300, activation="relu", kernel_initializer="he_normal")
		self.actor = Dense(81, activation="softmax") # 81 num acciones

	# Recorremos las redes de manera secuencial.
	def call(self, obs):
		ans = self.l1(obs)
		ans = self.actor(ans)
		return ans

class Critic(tf.keras.Model):

	# Utilizamos tensorflow.keras.layers.Dense para crear las redes.
	def __init__(self):
		super().__init__()
		self.l1 = Dense(300, activation="relu", kernel_initializer="he_normal")
		self.critic = Dense(1)

	# Recorremos las redes de manera secuencial.
	def call(self, obs):
		ans = self.l1(obs)
		ans = self.critic(ans)
		return ans

class ActorCriticAgent():

	def __init__(self):
		self.actor = Actor()
		self.critic = Critic()
		self.actorOpt = tf.keras.optimizers.Adam(learning_rate=0.0001)
		self.criticOpt = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.gamma = 0.99
		self.optLoad = False

	def act(self, obs):
		actionProb = self.actor(obs).numpy()
		action = tfp.distributions.Categorical(probs=actionProb, dtype=tf.float32)
		action = action.sample().numpy()[0]
		action = int(action)
		return action

	def getAction(self, sample):
		val = sample
		div = 81
		actions = []
		for i in range(4):
			div /= 3
			res = int(val / div)
			if res == 0: actions.append(0.0)
			elif res == 1: actions.append(1.0)
			else: actions.append(-1.0)
			val -= res * div
		return np.array(actions)

	# Se calcula la perdida en base a la ventaja de la accion y
	# el negativo del logaritmo de la probabilidad de la accion.
	def __actorLoss(self, probsIn, actions, advantage):
		probs = []
		logProbs = []
		# Calculamos los logaritmos de las probabilidades.
		for p, a in zip(probsIn, actions):
			dist = tfp.distributions.Categorical(probs=p, dtype=tf.float32)
			prob = dist.prob(a)
			logProb = dist.log_prob(a)
			probs.append(prob)
			logProbs.append(logProb)

		pLoss = []
		eLoss = []
		advantage = advantage.numpy()
		# Calculamos el negativo del logaritmo por la ventaja.
		# Tambien se calcula una entropia, a partir de la probabilidad
		# de la accion y su logaritmo.
		for p, adv, lp in zip(probs, advantage, logProbs):
			adv = tf.constant(adv)
			pl = tf.math.multiply(lp, adv)
			el = tf.math.negative(tf.math.multiply(p, lp))
			pLoss.append(pl)
			eLoss.append(el)
		pLoss = tf.stack(pLoss)
		eLoss = tf.stack(eLoss)
		pLoss = tf.reduce_mean(pLoss)
		eLoss = tf.reduce_mean(eLoss)
		# Agregamos la entropia a la perdida.
		loss = -pLoss - 0.01 * eLoss
		return loss

	# Es basicamente la recompensa descontada.
	def __expectedReturn(self, rewards):
		discRewards = []
		rewards.reverse()
		sumReward = 0
		for r in rewards:
			sumReward = r + self.gamma * sumReward
			discRewards.append(sumReward)
		discRewards.reverse()
		return discRewards

	# Calculamos los gradientes para cada red a partir de sus propias
	# perdidas.
	def train(self, states, actions, rewards):
		# Casteamos para que no haya problemas.
		states = np.array(states, dtype=np.float32)
		actions = np.array(actions, dtype=np.int32)
		# Recompensas descontadas.
		discRewards = self.__expectedReturn(rewards)
		discRewards = np.array(discRewards, dtype=np.float32)
		discRewards = tf.reshape(discRewards, (len(discRewards),))
		# Utilizamos tensorflow para calcular y aplicar gradientes.
		with tf.GradientTape() as tapeA, tf.GradientTape() as tapeC:
			tapeA.watch(discRewards)
			# Calculamos las probabilidades y el valor para la accion tomada.
			probs = self.actor(states, training=True)
			values = self.critic(states, training=True)
			values = tf.reshape(values, (len(values),))
			# Calculamos la ventaja.
			advantage = tf.math.subtract(discRewards, values)
			# Calculamos la perdida del actor.
			actorLoss = self.__actorLoss(probs, actions, advantage)
			# Calculamos la perdida del critico.
			criticLoss = 0.5 * tf.keras.losses.mean_squared_error(discRewards, values)
		# Calculamos los gradientes y optimizamos las redes utilizando Adam.
		gradsActor = tapeA.gradient(actorLoss, self.actor.trainable_variables)
		gradsCritic = tapeC.gradient(criticLoss, self.critic.trainable_variables)
		self.actorOpt.apply_gradients(zip(gradsActor, self.actor.trainable_variables))
		self.criticOpt.apply_gradients(zip(gradsCritic, self.critic.trainable_variables))

		if self.optLoad:
			self.loadOpts()

	def save(self, dir):
		self.actor.save_weights(dir + "_actor")
		self.critic.save_weights(dir + "_critic")
		with open(dir + "_optimizers.pickle", "wb") as f:
			pickle.dump([self.actorOpt.get_weights(), self.criticOpt.get_weights()], f)

	def load(self, dir):
		try:
			self.actor.load_weights(dir + "_actor")
		except:
			print("ERROR AL CARGAR ACTOR")
		try:
			self.critic.load_weights(dir + "_critic")
		except:
			print("ERROR AL CARGAR CRITIC")
		self.optDir = dir + "_optimizers.pickle"
		self.optLoad = True

	def loadOpts(self):
		try:
			with open(self.optDir, "rb") as f:
				actorOptWeights, criticOptWeights= pickle.load(f)
			self.actorOpt.set_weights(actorOptWeights)
			self.criticOpt.set_weights(criticOptWeights)
			print("CARGADOS LOS OPTIMIZER!")
		except:
			print("ERROR AL CARGAR LOS OPTIMIZER")
		self.optLoad = False

# https://keras.io/examples/rl/ddpg_pendulum/