import matplotlib.pyplot as plt
import pickle
import numpy as np

model = "DDPG"

dirDataDense = "./" + model + "_Dense_Save/" + model + "_rewards_info.pickle"
dirDataSparse = "./" + model + "_Sparse_Save/" + model + "_rewards_info.pickle"
with open(dirDataDense, "rb") as f:
	rewDense, _, succDense = pickle.load(f)
with open(dirDataSparse, "rb") as f:
	rewSparse, _, succSparse = pickle.load(f)

rewDense = [-50.0 if x >= -10.0 or x < -50.0 else x for x in rewDense]
rewSparse = [-50.0 if x >= -10.0 or x < -50.0 else x for x in rewSparse]
newRewDense = []
newSuccDense = []
newRewSparse = []
newSuccSparse = []

count = 0
tsumDense = [0, 0]
tsumSparse = [0, 0]

if len(rewDense) > len(rewSparse): xLen = len(rewDense)
else: xLen = len(rewSparse)

for i in range(xLen):
	if i < len(rewDense): tsumDense[0] += rewDense[i]
	else: tsumDense[0] += -50.0
	if i < len(succDense): tsumDense[1] += succDense[i]
	else: tsumDense[1] += 0.0
	if i < len(rewSparse): tsumSparse[0] += rewSparse[i]
	else: tsumSparse[0] += -50.0
	if i < len(succSparse): tsumSparse[1] += succSparse[i]
	else: tsumSparse[1] += 0.0

	if count % 10 == 0:
		if count == 0:
			newRewDense.append(tsumDense[0])
			newSuccDense.append(tsumDense[1])
			newRewSparse.append(tsumSparse[0])
			newSuccSparse.append(tsumSparse[1])

		else:
			newRewDense.append(tsumDense[0] / 10.0)
			newSuccDense.append(tsumDense[1] / 10.0)
			newRewSparse.append(tsumSparse[0] / 10.0)
			newSuccSparse.append(tsumSparse[1] / 10.0)

		tsumDense = [0, 0]
		tsumSparse = [0, 0]

	count += 1

x = np.arange(0, len(newRewDense)*1000, 1000)
'''
print(len(newRewDense), len(x))
plt.plot(x, newRewDense, label=model+" Dense")
plt.plot(x, newRewSparse, label=model+" Sparse")
plt.xlabel("Episode Nº")
plt.ylabel("Reward")
plt.legend()
plt.show()
'''
plt.plot(x, newSuccDense, label=model+" Dense")
plt.plot(x, newSuccSparse, label=model+" Sparse")
plt.xlabel("Episode Nº")
plt.ylabel("Success Ratio")
plt.legend()
plt.show()
