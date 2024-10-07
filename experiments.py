import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2 import GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model : GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained("gpt2")
# set model to evaluation mode
model.eval()

text = "Who are you?"
outputTokens = []
with torch.no_grad():
	input_ids = tokenizer(text, return_tensors="pt").input_ids
	for i in range(100):
		output = model(input_ids, output_hidden_states=False)
		outputTokens.append(output.logits[:, -1, :].argmax())
		input_ids = torch.cat([input_ids, torch.tensor([[outputTokens[-1]]])], dim=-1)

print(tokenizer.decode(input_ids[0]))
