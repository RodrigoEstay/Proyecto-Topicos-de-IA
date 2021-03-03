import gym
import numpy as np
from gym import wrappers
import glfw
from types import MethodType
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sys import exit
from typing import Any, List, Sequence, Tuple
import os

from DDPG import DDPG
import pickle

def env_wrapper(env):
	def close(self):
		if self.viewer is not None:
			glfw.destroy_window(self.viewer.window)
			self.viewer = None

	env.unwrapped.close = MethodType(close, env.unwrapped)
	return env

def preprocess(obs):
	newObs = []
	newObs.extend(obs["observation"])
	#newObs.extend(obs["achieved_goal"])
	newObs.extend(obs["desired_goal"])
	return np.array([newObs])

saveDir = "./DDPG_Dense_Save/"
agent = DDPG()
episodes = 1000000
env = env_wrapper(gym.make('FetchSlide-v1'))

if not os.path.isdir(saveDir):
	os.mkdir(saveDir)

lastEp = 0
rewardsPerSave = []
succesRatioPerSave = []
totalRewards = []
totalSuccess = 0

epsPerSave = 100
render = True

infoDir = saveDir + "DDPG_"
if os.path.isfile(infoDir + "rewards_info.pickle"):
	with open(infoDir + "rewards_info.pickle", "rb") as f:
		rewardsPerSave, lastEp, succesRatioPerSave = pickle.load(f)
	agent.load(infoDir)
	print("Cargado save anterior!")


for t in range(lastEp, episodes):
	prevObservation = env.reset()
	initDist = np.sqrt(sum(pow(prevObservation["achieved_goal"] - prevObservation["desired_goal"], 2)))
	prevObservation = preprocess(prevObservation)
	done = False

	totalReward = 0
	
	while not done:
		if render:
			try:
				env.render()
			except NameError:
				env.close()
				print("CERRADO MANUAL")
				exit()

		action = agent.act(prevObservation)
		observation, reward, done, info = env.step(action)
		newDist = np.sqrt(sum(pow(observation["achieved_goal"] - observation["desired_goal"], 2)))
		observation = preprocess(observation)
		if reward != 0.0:
			reward += (initDist - newDist) / initDist
			if reward < -1.0: reward = -1.0
		agent.record(prevObservation, action, reward, observation)
		prevObservation = observation

		totalReward += reward
	
	agent.train()
	if info["is_success"] != 0.0:
		totalSuccess += 1
	totalRewards.append(totalReward)

	if t % epsPerSave == 0:
		rew = sum(totalRewards) / epsPerSave
		succesRatio = totalSuccess / epsPerSave
		print("EPISODIO {} REWARD PROMEDIO: {}, SUCCES RATIO: {}".format(t, rew, succesRatio))

		rewardsPerSave.append(rew)
		succesRatioPerSave.append(succesRatio)

		with open(infoDir + "rewards_info.pickle", "wb") as f:
			pickle.dump([rewardsPerSave, t, succesRatioPerSave], f)
		agent.save(infoDir)

		totalRewards = []
		totalSuccess = 0

env.close()

# https://towardsdatascience.com/reinforce-policy-gradient-with-tensorflow2-x-be1dea695f24
# https://towardsdatascience.com/building-our-first-neural-network-in-keras-bdc8abbc17f5