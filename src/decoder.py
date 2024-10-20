from collections.abc import Callable
from typing import Any, List, Optional, Tuple
from overrides import override
from numpy import ndarray as NDArray
from torch import Tensor
import numpy as np
import torch

from src.utils import getMostProbableToken, getTopKTokens

class Decoder:
    def __init__(self):
        pass

    def step(*args, **kwargs) -> Any:
        raise NotImplementedError

    def decode(*args, **kwargs) -> Any :
        raise NotImplementedError

class SimpleDecoder(Decoder):
    def __init__(self, model, config, samplingScheme : Optional[Callable] = None):
        super().__init__()
        self.model = model
        self.config = config

        if samplingScheme is None :
            samplingScheme = getMostProbableToken

        self.sampler = samplingScheme
    
    @override
    def step(self, inputSeq : Tensor, maxLen : int, *args, **kwargs) -> Any:
        """
        Passes the inputSeq to the model and 
        decodes the next token.
        """
        # return self.decode(self.model.infer(inputSeq), *args, **kwargs)
        for _ in range(maxLen):
            distribution = self.model.infer(inputSeq)
            index = self.decode(distribution, *args, **kwargs)
            inputSeq = torch.cat((inputSeq, index.unsqueeze(1)), dim = 1)
        return inputSeq[..., -maxLen:]
    
    @override
    def decode(self, *args, **kwargs) -> Any :
        """
        The method takes in the distribution from the 
        model and returns the most likely token by 
        aplying the decoding scheme.
        """
        return self.sampler(*args, **kwargs)

class BeamSearchDecoder(Decoder):
    def __init__(self, model, beamSize : int, config : dict, samplingScheme : Optional[Callable] = None):
        """
        Implements the Beam Search Decoding scheme.
        @params beamSize : The size of the beam.
        @params config   : The configuration of the model. 
                           In beam search, the config must 
                           contain the index of the EOS token.
        """
        super().__init__()
        self.model = model
        self.config = config
        self.beamSize = beamSize
        if samplingScheme is None : 
            samplingScheme = getTopKTokens

        self.sampler = samplingScheme
    
    @override
    def step(self, inputSeq : NDArray, numTokens : int = 1, *args, **kwargs ) -> List :
        """
        Takes in the input sequence and generates the
        next token.
        @param inputSeq  : The input sequence ( Not Batched ).
        @param numTokens : Number of tokens to decode, by default
                           the value is 1.
        """
        inputSeq = np.array(inputSeq)
        beams : List[Tuple[List[int], float]] = [(inputSeq.copy(), 0)]

        for _ in range(numTokens) :
            newCandidates : List[Tuple[List[int], float]] = []

            for ( sequences, logProbs ) in beams :
                # print(sequences)
                distribution = self.model.infer(torch.tensor(sequences))
                distribution = np.array(distribution)
                topKIndices, topKValues = self.decode(distribution, *args, normalize = False, **kwargs)

                for index, value in zip(topKIndices[0], topKValues[0]) :
                    # print(sequences, index)
                    newSeq = np.concatenate((sequences, [[index]]), axis = -1)
                    # print(newSeq)
                    newLogProb = logProbs + np.log(max(value, 1e-10))
                    newCandidates.append((newSeq, newLogProb))

            newCandidates.sort(key = lambda x : x[1], reverse = True)
            beams = newCandidates[:self.beamSize]

        return list(beams[0][0][...,-numTokens:])

    @override
    def decode(self, distribution : NDArray, *args, **kwargs) -> Tuple[NDArray, NDArray] :
        """
        @param distribution : The distribution from the
        model that we want to decode. Should be of the size 
        of the vocab.
        """
        return self.sampler(distribution, k = self.beamSize, *args, **kwargs)
