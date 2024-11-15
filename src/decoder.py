from collections.abc import Callable
from typing import Any, List, Optional, Tuple
from overrides import override
from torch import Tensor
import torch

from src.utils import getMostProbableToken, getTopKTokens

EPSILON = 1e-10

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
    def step(self, inputSeq : Tensor, numTokens : int = 1, *args, **kwargs ) -> List :
        """
        Takes in the input sequence and generates the
        next token.
        @param inputSeq  : The input sequence ( Not Batched ).
        @param numTokens : Number of tokens to decode, by default
                           the value is 1.
        """
        beams : List[Tuple[Tensor, float]] = [(inputSeq.clone(), 0)]

        for _ in range(numTokens) :
            newCandidates : List[Tuple[Tensor, float]] = []

            for ( sequences, logProbs ) in beams :
                # print(sequences)
                distribution = self.model.infer(torch.tensor(sequences))
                topKIndices, topKValues = self.decode(distribution, *args, normalize = False, **kwargs)

                for index, value in zip(topKIndices[0], topKValues[0]) :
                    # print(sequences, index)
                    newSeq = torch.cat((sequences, torch.tensor([[index]])), dim=-1)
                    # print(newSeq)
                    newLogProb = logProbs + torch.log(value+1e-10)
                    newCandidates.append((newSeq, newLogProb.item()))

            newCandidates.sort(key = lambda x : x[1], reverse = True)
            beams = newCandidates[:self.beamSize]
        
        return list(beams[0][0][...,-numTokens:])

    @override
    def decode(self, distribution : Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor] :
        """
        @param distribution : The distribution from the
        model that we want to decode. Should be of the size 
        of the vocab.
        """
        return self.sampler(distribution, k = self.beamSize, *args, **kwargs)

class SpeculativeDecoder(Decoder):
    def __init__(self, model, k : int, draftModelDecoder, samplingScheme : Optional[Callable] = None, **kwargs):
        super().__init__()
        self.model = model
        self.k = k
        self.draftModelDecoder = draftModelDecoder
        if samplingScheme is None : 
            samplingScheme = getTopKTokens
        
        self.samplingKwargs = kwargs
        self.sampler = samplingScheme
    
    @override
    def step(self, inputSeq : Tensor, numTokens : int = 5, *args, **kwargs ) -> tuple[Tensor,Tensor] :
        """
        @param inputSeq  : The input sequence (Not Batched).
        @param numTokens : Number of tokens to decode, by default
                           the value is 5.
        """
        additionalTokens = []
        
        while len(additionalTokens) < numTokens :
            nextTokens, probabilityDistributions = self.decode(inputSeq, samplingKwargs = self.samplingKwargs, *args, **kwargs)
            additionalTokens.extend([(token,probability) for token, probability in zip(nextTokens, probabilityDistributions)])
            inputSeq = torch.cat((inputSeq, nextTokens.unsqueeze(1)), dim = 1)
        
        additionalTokens = additionalTokens[:numTokens]

        if len(additionalTokens) == 0 :
            print("No additional tokens generated")
            return torch.tensor([]), torch.tensor([])
    
        additionalTokenIds, tokenProbabiltiyDistributions = zip(*additionalTokens)

        return torch.stack(additionalTokenIds), torch.stack(tokenProbabiltiyDistributions)

    @override
    def decode(self, inputSeq : Tensor , samplingKwargs,  *args, **kwargs) -> tuple[Tensor,Tensor] :
        """
        @param inputSeq  : The input sequence.
        """
        additionalTokens = []

        # First call the smaller model to get the additional k tokens
        # Along with their probabilities.
        draftInput : Tensor = self.draftModelDecoder.step(numTokens = self.k, inputSeq = inputSeq, *args, **kwargs)
        
        draftModelPredictions = draftInput[0]
        draftModelDistributions = draftInput[1]

        # Now we call the larger model to get the outputs.
        modelOutputPredictions : Tensor = self.model.infer(torch.cat((inputSeq, draftModelPredictions), dim = -1), lastK = self.k + 1)

        for i in range(self.k) :
            # p(x)
            modelOutputDistribution : Tensor = modelOutputPredictions[i]
            # q(x)
            draftModelDistribution : Tensor = draftModelDistributions[i]

            # output by the draft model
            draftModelToken : Tensor = draftModelPredictions[i]
            
            draftModelProbability : Tensor = draftModelDistribution[draftModelToken]
            modelOutputProbability : Tensor = modelOutputDistribution[draftModelToken]
            
            if modelOutputProbability >= draftModelProbability :
                # The draft is fine and we can move on.
                # ACCEPTED
                additionalTokens.append((draftModelToken, modelOutputDistribution))
            else :
                # We reject the draft token with probability 1 - modelOutputProbability/draftModelProbability
                rejectionProbability = torch.clamp(1 - modelOutputProbability / draftModelProbability, min=0, max=1)
                rejected = torch.bernoulli(rejectionProbability)

                if rejected : 
                    # REJECTED
                    # In this case we sample from the normalized and adjusted probability distribution :
                    # p'(x) = norm(max(0, p(x) - q(x)))
                    adjustedProbabilityDistribution = torch.max(torch.zeros_like(modelOutputDistribution), modelOutputDistribution - draftModelDistribution)

                    adjustedProbabilityDistribution = adjustedProbabilityDistribution / (adjustedProbabilityDistribution.sum(dim = -1, keepdim = True) + EPSILON)
                    
                    # Sample from the adjusted probability distribution
                    nextToken = torch.multinomial(adjustedProbabilityDistribution, num_samples = 1)[0]
                    additionalTokens.append((nextToken, adjustedProbabilityDistribution))
                    break
                else :
                    # ACCEPTED
                    additionalTokens.append((draftModelToken, draftModelDistribution))

        # Check the number of additional tokens generated 
        if len(additionalTokens) == self.k :
            # All draft tokens were accepted.
            # In this case, we will add the final token 
            # from the larger model.
            additionalTokens.append((self.sampler(modelOutputPredictions[..., -1], **samplingKwargs)[0], modelOutputPredictions[..., -1]))
        
        additionalTokenIds, additionalTokenDistributions = zip(*additionalTokens)

        return torch.stack(additionalTokenIds), torch.stack(additionalTokenDistributions)
