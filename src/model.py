from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")	


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
			return self.model(inputSeq, output_hidden_states=False).logits[0, -lastK:, :]
