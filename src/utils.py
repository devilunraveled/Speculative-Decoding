from numpy import ndarray as NDArray
import random

############ SAMPLING TECHNIQUES ############

def getMostProbableToken( distribution : NDArray ) :
    """
    Returns the token index with the highest probabilty. 
    """
    return distribution.argmax(axis = -1)

def getATokenFromTopK( distribution : NDArray, k : int ) :
    """
    Returns a weighted random choice from the normalized
    probability distribution from the top k most 
    probable tokens.
    """
    topKIndices = distribution.argsort(axis=-1)[-k:]
    topKValues = distribution[topKIndices]
    return random.choices(population=topKIndices, weights=topKValues)

def getTopKTokens( distribution : NDArray , k : int) :
    """
    Returns the top k most probable tokens.
    """
    return getATokenFromTopK(distribution, k)
