import numpy as np
from numpy import ndarray as NDArray
import numpy.random as random

############ SAMPLING TECHNIQUES ############

def getMostProbableToken( distribution : NDArray ) :
    """
    Returns the token index with the highest probabilty. 
    """
    return distribution.argmax(axis = -1)

def getATokenFromTopK( distribution : NDArray, k : int) :
    """
    Returns a weighted random choice from the normalized
    probability distribution from the top k most 
    probable tokens.
    """
    topKIndices, topKValues = getTopKTokens(distribution, k)
    if topKValues.sum(axis = -1, dtype = 'float32') == 0 :
        raise Exception("All values are 0.")

    topKValues = topKValues / topKValues.sum(axis = -1, dtype = 'float32')
    return random.choice(a = topKIndices, p = topKValues, replace = False)

def getTopKTokens( distribution : NDArray , k : int, normalize : bool = False ) :
    """
    Returns the top k most probable tokens.
    """
    topKIndices = distribution.argsort(axis=-1)[..., -k:]
    topKValues = np.take_along_axis(distribution, topKIndices, axis=-1)
    if normalize :
        topKValues = topKValues / topKValues.sum(axis = -1, dtype = 'float32')
    
    return topKIndices, topKValues
