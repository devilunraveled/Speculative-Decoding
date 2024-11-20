from collections.abc import Callable
from typing import Any, List, Optional, Tuple
from overrides import override
import torch

from src.utils import getMostProbableToken, getTopKTokens, Information

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
    def step(self, inputSeq , numTokens : int, *args, **kwargs) -> Any:
        """
        Passes the inputSeq to the model and 
        decodes the next token.
        """
        # return self.decode(self.model.infer(inputSeq), *args, **kwargs)
        information = Information(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, None)
        additionalTokens = []

        for _ in range(numTokens):
            distribution = self.model.infer(inputSeq)
            information.max_util = max(information.max_util, torch.cuda.utilization())
            index = self.decode(distribution, *args, **kwargs)[0]
            additionalTokens.append((index, distribution))
            inputSeq = torch.cat((inputSeq, index.unsqueeze(1)), dim = 1)
        
        additionalTokenIds, additionalTokenDistributions = zip(*additionalTokens)
        return torch.cat(additionalTokenIds), torch.cat(additionalTokenDistributions), information
    
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
    def step(self, inputSeq, numTokens : int = 1, *args, **kwargs ) -> List :
        """
        Takes in the input sequence and generates the
        next token.
        @param inputSeq  : The input sequence ( Not Batched ).
        @param numTokens : Number of tokens to decode, by default
                           the value is 1.
        """

        information = Information(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, None)

        beamOutput = self.model.model.generate(
                                            inputSeq,
                                            max_length = inputSeq.shape[1] + numTokens,
                                            num_beams = self.beamSize,
                                            output_logits = True,
                                            output_scores = True,
                                            return_dict_in_generate = True,
                                            )
        information.max_util = max(information.max_util, torch.cuda.utilization())
        information.memory_footprint = max(information.memory_footprint, torch.cuda.memory_allocated() / 1024 / 1024)
        logits = beamOutput.logits[0][beamOutput.beam_indices[0]][len(inputSeq[0]):]
        distribution = torch.softmax(logits, dim = -1)
        return beamOutput.sequences[0][len(inputSeq[0]):], distribution, information

    @override
    def decode(self, distribution , *args, **kwargs) :
        """
        @param distribution : The distribution from the
        model that we want to decode. Should be of the size 
        of the vocab.
        """
        return self.sampler(distribution, k = self.beamSize, *args, **kwargs)

class SpeculativeDecoder(Decoder):
    def __init__(self, model, gamma : int, draftModelDecoder, samplingScheme : Optional[Callable] = None, **kwargs):
        super().__init__()
        self.model = model
        self.k = gamma
        self.draftModelDecoder = draftModelDecoder
        if samplingScheme is None :
            print("Using default sampling scheme: greedy")
            samplingScheme = getMostProbableToken
        
        self.samplingKwargs = kwargs
        self.sampler = samplingScheme
    
    @override
    def step(self, inputSeq , numTokens : int = 5, *args, **kwargs ) :
        """
        @param inputSeq  : The input sequence (Not Batched).
        @param numTokens : Number of tokens to decode, by default
                           the value is 5.
        """

        information = Information(0, 0, 0, 0, 0, 0.0, 0.0, 0.0)
        
        additionalTokens = []
        
        while len(additionalTokens) < numTokens :
            nextTokens, probabilityDistributions, information = self.decode(inputSeq, samplingKwargs = self.samplingKwargs, *args, information = information, **kwargs)
            additionalTokens.extend([(token,probability) for token, probability in zip(nextTokens, probabilityDistributions)])
            inputSeq = torch.cat((inputSeq, nextTokens.unsqueeze(0)), dim = 1)

        additionalTokens = additionalTokens[:numTokens]

        if len(additionalTokens) == 0 :
            print("No additional tokens generated")
            return torch.tensor([]), torch.tensor([]), information
    
        additionalTokenIds, tokenProbabiltiyDistributions = zip(*additionalTokens)

        # print(f'Decoded from {self.model.model.name_or_path} : {additionalTokenIds}')
        # print(f'Probabilities : {tokenProbabiltiyDistributions}')

        return torch.stack(additionalTokenIds), torch.stack(tokenProbabiltiyDistributions), information

    @override
    def decode(self, inputSeq , samplingKwargs,  *args, **kwargs) :
        """
        @param inputSeq  : The input sequence.
        """
        verbose = kwargs.pop("verbose", False)
        information = kwargs.pop("information")
        additionalTokens = []

        # print(f"Input Seq Shape : {inputSeq.shape}")
        # First call the smaller model to get the additional k tokens
        # Along with their probabilities.
        draftInput = self.draftModelDecoder.step(numTokens = self.k, inputSeq = inputSeq, *args, **kwargs)
        information.max_util = max(information.max_util, torch.cuda.utilization())
        information.memory_footprint = max(information.memory_footprint, torch.cuda.memory_allocated() / 1024 / 1024)
        
        draftModelPredictions = draftInput[0]
        # print(f"Draft Model Predictions Shape : {draftModelPredictions.shape}")
        draftModelDistributions = draftInput[1]
        # print(f"Draft Model Distribution Shape : {draftModelDistributions.shape}")
        information.subDecoder_info = draftInput[2]
        
        information.drafted += self.k # Total Proposed

        # Now we call the larger model to get the outputs.
        modelOutputPredictions = self.model.infer(torch.cat((inputSeq, draftModelPredictions.unsqueeze(0)), dim = -1), lastK = self.k + 1)
        information.max_util = max(information.max_util, torch.cuda.utilization())
        information.memory_footprint = max(information.memory_footprint, torch.cuda.memory_allocated() / 1024 / 1024)
        # print(f"Model Output Predictions Shape : {modelOutputPredictions.shape}")

        for i in range(self.k) :
            # p(x)
            modelOutputDistribution  = modelOutputPredictions[i]
            # q(x)
            draftModelDistribution  = draftModelDistributions[i]

            # output by the draft model
            draftModelToken  = draftModelPredictions[i]
            if verbose :
                print(f"Draft Model Token : `{self.model.tokenizer.decode(draftModelToken, skip_special_tokens = True)}`")
            
            draftModelProbability  = draftModelDistribution[draftModelToken]
            modelOutputProbability  = modelOutputDistribution[draftModelToken]
            
            if modelOutputProbability >= draftModelProbability :
                # The draft is fine and we can move on.
                # ACCEPTED
                information.accept_direct += 1
                if verbose :
                    print(f"Token : `{self.model.tokenizer.decode(draftModelToken, skip_special_tokens = True)}` Accepted directly.")
                additionalTokens.append((draftModelToken, modelOutputDistribution))
            else :
                # We reject the draft token with probability 1 - modelOutputProbability/draftModelProbability
                rejectionProbability = torch.clamp(1 - modelOutputProbability / draftModelProbability, min=0, max=1)
                rejected = torch.bernoulli(rejectionProbability)

                if rejected : 
                    # REJECTED
                    information.reject += (self.k - i)
                    # In this case we sample from the normalized and adjusted probability distribution :
                    # p'(x) = norm(max(0, p(x) - q(x)))
                    adjustedProbabilityDistribution = torch.clip(modelOutputDistribution - draftModelDistribution, min = 0, max = 1)

                    adjustedProbabilityDistribution = adjustedProbabilityDistribution / (adjustedProbabilityDistribution.sum(dim = -1, keepdim = True) + EPSILON)
                    
                    # Sample from the adjusted probability distribution
                    nextToken = self.sampler(adjustedProbabilityDistribution, **samplingKwargs)[0]
                    if verbose :
                        print(f"Draft Token Rejected. Next Token : `{self.model.tokenizer.decode(nextToken, skip_special_tokens = True)}`")
                    additionalTokens.append((nextToken, adjustedProbabilityDistribution))
                    break
                else :
                    # ACCEPTED
                    information.accept_indirect += 1
                    if verbose :
                        print(f"Token : {self.model.tokenizer.decode(draftModelToken, skip_special_tokens = True)} Accepted.")
                    additionalTokens.append((draftModelToken, draftModelDistribution))

        # Check the number of additional tokens generated 
        if len(additionalTokens) == self.k :
            # All draft tokens were accepted.
            # In this case, we will add the final token 
            # from the larger model.
            targetModelToken = self.sampler(modelOutputPredictions[-1], **samplingKwargs)[0]
            if verbose :
                print(f"Final Token from Target Model : `{self.model.tokenizer.decode(targetModelToken, skip_special_tokens = True)}`")
            additionalTokens.append((targetModelToken, modelOutputPredictions[-1]))
        
        additionalTokenIds, additionalTokenDistributions = zip(*additionalTokens)
        information.total_generated += len(additionalTokenIds)

        return torch.stack(additionalTokenIds), torch.stack(additionalTokenDistributions), information
