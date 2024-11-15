from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")	

from typing import List
import torch

from .decoder import Decoder as BaseDecoder

class HuggingFaceModelWrapper:
	def __init__(self, model):
		self.model = model
		self.model.eval()
	
	def infer(self, inputSeq, lastK = 1):
		"""
		:param inputSeq: a (batched) sequence of tokens.
		:return: a (batched) distribution over the vocabulary.
		"""
		with torch.no_grad():
			return self.model(inputSeq, output_hidden_states=False).logits[:, -lastK:, :]
	
class Inferencer:
	def __init__(self, tokenizer, decoder : BaseDecoder):
		self.tokenizer = tokenizer
		self.decoder = decoder
	
	def decode(self, inputSeq : List[int], maxLen : int = 100) -> List[int]:
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
		output = self.decode(tokens, maxLen)[0]
		return self.tokenizer.decode(torch.tensor(output))
