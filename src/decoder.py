from collections.abc import Callable
from typing import Any, List, Optional, override
from numpy.typing import NDArray

from utils import getMostProbableToken

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
    def step(self, inputSeq, *args, **kwargs) -> Any:
        """
        Passes the inputSeq to the model and 
        decodes the next token.
        """
        return self.decode(self.model.infer(inputSeq), *args, **kwargs)
    
    @override
    def decode(self, *args, **kwargs) -> Any :
        """
        The method takes in the distribution from the 
        model and returns the most likely token by 
        aplying the decoding scheme.
        """
        return self.sampler(*args, **kwargs)

class BeamSearchDecoder(Decoder):
    def __init__(self, model, beamSize : int, config : dict, samplingScheme : Optional[Callable[[NDArray], List[int]]] = None):
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
        self.sampler = samplingScheme
    
    @override
    def step(self, inputSeq : List[int], numTokens : int = 1 ) -> int :
        """
        Takes in the input sequence and generates the
        next token.
        @param inputSeq  : The input sequence ( Not Batched ).
        @param numTokens : Number of tokens to decode, by default
                           the value is 1.
        """
    
    @override
    def decode(self, distribution : NDArray) -> int:
        """
        @param distribution : The distribution from the
        model that we want to decode. Should be of the size 
        of the vocab.
        """
