import mountaincar
import numpy
from Tilecoder import numTilings, tilecode, numTiles
from pylab import *

numRuns = 1
numEpisodes = 300
alpha = 0.05/numTilings
gamma = 1
lmbda = 0.9
epsilon = 0
n = numTiles * 3
# eligibility traces
eligibilityTraces = zeros(n)
F = [-1]*numTilings
terminalPosition = 0.5

# Calculate the state action value by adjusting the feature indices for the current action
def getStateActionValue(weights, features, action):
	stateActionValue = 0
	for i in range(numTilings):
		tempFeatureIndex = features[i] + (numTiles * action)
		stateActionValue += weights[tempFeatureIndex]
	return stateActionValue
		
# Calculate the epsilon greedy action at the current state
def getEpsilonGreedyAction(Q):
	# epsilon must be between 0.0 and 1.0
	assert (epsilon >= 0.0) and (epsilon <= 1.0)
	# Choose greedy action with probability 1 - epsilon
	action = getGreedyAction(Q)
	# Choose random action with probability epsilon
	sample = numpy.random.random_sample()
	if sample <= epsilon:
		action = numpy.random.randint(0,3)
	return action
	
# Calculate and return the greedy action at the current state
def getGreedyAction(Q):
	# Find action at current state, that has the max action value
	greedyAction = argmax(Q)
	# If all actions have the same action value, choose action at random
	if Q[0] == Q[1] == Q[2]:
		greedyAction = numpy.random.randint(0,3)
	return greedyAction	

# Update the weights using the eligibility traces
def updateWeights(weights, delta):
	tempTraces = numpy.multiply(alpha * delta, eligibilityTraces)
	return numpy.add(w, tempTraces)
	

runSum = 0.0
for run in range(numRuns):
    w = -0.01*rand(n)
    returnSum = 0.0
    # state-action values for each action at the current state
    q = zeros(3)
    for episodeNum in range(numEpisodes):
        #print('Episode: ' + str(episodeNum))
        G = 0
        pos, vel = mountaincar.init()
        # For each step of the episode
        while (pos < terminalPosition):
        	#print('Position: ' + str(pos))
			# Initialize state action values
        	Q = zeros(3)
        	for a in [0,1,2]:
        		tilecode(pos, vel, F)
        		# find the state action value at each state using the weights and the 
        		# active feature indices
        		Q[a] = getStateActionValue(w, F, a)
        	# Choose the action that give the max state action value using epsilon greedy
        	action = getEpsilonGreedyAction(Q)
        	print(str(pos) + ', ' + str(action))
        	# take action and observe reward and next state
        	R, (newPos, newVel) = mountaincar.sample((pos, vel), action)
        	delta = R - Q[action]
        	# Update the eligibility traces for current state and all actions
        	for actionIndex in range(3):
        		for featureIndex in range(numTilings):
        			tempFeatureIndex = featureIndex + (numTiles * actionIndex)
        			eligibilityTraces[tempFeatureIndex] = 1
        	# If new state is terminal, go to next episode
        	if (pos >= 0.5):
        		w = updateWeights(w, delta)
        		break;
        	# Get the new state action values for the newly observed state
        	Q = zeros(3)
        	for a in [0,1,2]:
        		tilecode(newPos, newVel, F)
        		# find the state action value at each state using the weights and the 
        		# active feature indices
        		Q[a] = getStateActionValue(w, F, a)
        	# Update delta, weight vector, eligibility traces, and current state
        	delta = delta + gamma * max(Q)
        	w = updateWeights(w, delta)
        	eligibilityTraces = numpy.multiply(gamma * lmbda, eligibilityTraces)
        	pos = newPos
        	vel = newVel
        print("Episode: ", episodeNum, "Return: ", G)
        returnSum = returnSum + G
    print("Average return:", returnSum/numEpisodes)
    runSum += returnSum
print("Overall average return:", runSum/numRuns/numEpisodes)

def writeF():
    fout = open('value', 'w')
    F = [0]*numTilings
    steps = 50
    for i in range(steps):
        for j in range(steps):
            tilecode(-1.2+i*1.7/steps, -0.07+j*0.14/steps, F)
            height = -max(Qs(F))
            fout.write(repr(height) + ' ')
        fout.write('\n')
    fout.close()

def writeAverages(filename,averages):
    savetxt(filename,averages)
