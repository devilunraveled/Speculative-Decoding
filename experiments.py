import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2 import GPT2LMHeadModel

from src.decoder import GreedyDecoder

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model : GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained("gpt2")
# set model to evaluation mode
model.eval()

text = "Who are you?"
with torch.no_grad():
	# get next token logits
	input_ids = tokenizer(text, return_tensors="pt").input_ids
	output = model(input_ids, output_hidden_states=False)

print(output.logits[:, -1, :])
print(tokenizer.vocab_size)

decoder = GreedyDecoder(model, None)