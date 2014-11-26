numTilings = 4
oneDimensionTileCount = 9
numTiles = oneDimensionTileCount * oneDimensionTileCount * numTilings
# position tile size if (maxPosition - minPosition) / (oneDimensionTileCount - 1)
positionTileSize = 0.2125
minPosition = -1.2
maxPosition = 0.5
# position tile size if (maxVelocity - minVelocity) / (oneDimensionTileCount - 1)
velocityTileSize = 0.0175
minVelocity = -0.07
maxVelocity = 0.07

def tilecode(S, tileIndices):
	# Unpack S
	assert (len(S) == 2)
	pos = S[0]
	vel = S[1]

	# Preconditions
	assert (pos >= minPosition) and (pos <= maxPosition)
	assert (vel >= minPosition) and (vel <= maxPosition)
	# Calculate the indicies in each of the tilings
	for tilingIndex in range(numTilings):
		# get local tile coordinates 
		tileX = int( (pos + (tilingIndex*positionTileSize)/numTilings) / positionTileSize)
		tileY = int( (vel + (tilingIndex*velocityTileSize)/numTilings) / velocityTileSize)

		# get the local tile number (between 0 and oneDimensionTileCount^2)
		localTileNumber = tileX + tileY*(oneDimensionTileCount)

		# get the global tile number and populate the tileIndices vector
		globalTileNumber = tilingIndex*(pow(oneDimensionTileCount,2)) + localTileNumber
		tileIndices[tilingIndex] = globalTileNumber

def printTileCoderIndices(pos, vel):
    tileIndices = [-1]*numTilings
    tilecode(pos, vel, tileIndices)
    print('Tile indices for input (', pos,',', vel,') are : ', tileIndices)

#printTileCoderIndices(0.1,0.1)
#printTileCoderIndices(4.0,2.0)
#printTileCoderIndices(5.99,5.99)
#printTileCoderIndices(4.0,2.1)
    
