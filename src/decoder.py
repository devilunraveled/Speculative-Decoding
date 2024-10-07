from typing import Any
from numpy.typing import NDArray

class Decoder:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def decode(self, *args, **kwargs) -> Any :
        """
        The method takes in the distribution from the 
        model and returns the most likely token by 
        aplying the decoding scheme.
        """
        raise NotImplementedError

class GreedyDecoder(Decoder):
    def __init__(self, model, config):
        super().__init__(model, config)

    def decode(self, logits):
        return logits.argmax(dim=-1)

class TopKDecoder(Decoder):
    def __init__(self, model, k, config):
        super().__init__(model, config)
        self.k = k

    def decode(self, distribution : NDArray) -> int:
        topKSamples = distribution.argsort(axis=-1)[-self.k:]
        # Re-normalize the distribution.
        topKSamples = topKSamples / topKSamples.sum(axis=-1, keepdims=True)
        # Pick the sample with the highest probability.
        return topKSamples

class BeamSearchDecoder(Decoder):
    def __init__(self, model, beamSize, config):
        super().__init__(model, config)
        self.beamSize = beamSize
