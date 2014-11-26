import mountaincar
import numpy
from Tilecoder_v2 import numTilings, tilecode, numTiles
from pylab import *

numRuns = 1
numEpisodes = 300
n = numTiles * 3

alpha = 0.05/numTilings
gamma = 1
lmbda = 0.9
epsilon = 0


# Calculate the state action value by adjusting the feature indices for the current action
def getStateActionValue(weights, features, action):
    stateActionValue = 0
    # the list of feature indices must be of the correct length
    assert (len(features) == numTilings)
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

runSum = 0.0
for runNum in range(numRuns):
    returnSum = 0.0
    w = zeros(n)
    for episodeNum in range(numEpisodes):
        G = 0
        e = zeros(n)
        carState = mountaincar.init()
        while not carState==None:
            Qa = zeros(3)
            Fa = zeros(4)
            for a_poss in [0,1,2]:
                tilecode(carState,Fa)
                assert (sum(Fa) > 0) # make sure Fa is populated
                Qa[a_poss] = getStateActionValue(w,Fa,a_poss)

            # get an action, act on it, and observe the reward
            A = getEpsilonGreedyAction(Qa)
            R,carStateNew = mountaincar.sample(carState,A)
            G = G + R

            delta = R - Qa[A]

            for i in Fa: # for each active feature index
                e[i+numTiles*A] = 1

            # if the new state is the terminal state, update the weight vector and break
            if carStateNew==None:
                w = w + alpha*delta * e
                break

            # update values for the weight vector and the eligibility traces
            Qa = zeros(3)
            Fa = zeros(4)
            for a_poss in [0,1,2]:
                tilecode(carStateNew,Fa)
                Qa[a_poss] = getStateActionValue(w,Fa,a_poss)           
            delta = delta + gamma*max(Qa)
            
            w = w + alpha*delta * e
            e = gamma*lmbda*e

            # move to the new state
            carState = carStateNew
        print("Episode: ", episodeNum, "Return: ", G)
        returnSum = returnSum + G
    print("Average return:", returnSum/numEpisodes)
    runSum += returnSum
print("Overall average return:", runSum/numRuns/numEpisodes)