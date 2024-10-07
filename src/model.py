from typing import List

from .decoder import Decoder as BaseDecoder

class HuggingFaceModelWrapper:
	def __init__(self, model, tokenizer):
		self.model = model
		self.model.eval()

		self.tokenizer = tokenizer
	
	def infer(self, inputSeq):
		"""
		:param inputSeq: a (batched) sequence of tokens.
		:return: a (batched) distribution over the vocabulary.
		"""
		return self.model(inputSeq, output_hidden_states=False).logits[:, -1, :]
	
class Inferencer:
	def __init__(self, tokenizer, decoder : BaseDecoder):
		self.tokenizer = tokenizer
		self.decoder = decoder
	
	def decode(self, inputSeq : List[int], maxLen : int = 100) -> str:
		"""
		:param inputSeq: a (batched) sequence of tokens.
		:return: a (batched) distribution over the vocabulary.
		"""
		return self.decoder.step(inputSeq, maxLen)
	
	def generate(self, inputText : str, maxLen : int = 100) -> str:
		"""
		:param inputText: prompt to be fed into the model
		:return outputText: generated text
		"""
		tokens = self.tokenizer(inputText, return_tensors="pt").input_ids
		output = self.decode(tokens, maxLen)