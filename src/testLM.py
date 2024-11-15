from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer = AutoTokenizer.from_pretrained("amd/AMD-Llama-135m")
# model = AutoModelForCausalLM.from_pretrained("amd/AMD-Llama-135m")

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")

prompt = "Hi how are you?"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model(input_ids, output_hidden_states=False).logits[:, -3:, :]
print(output.shape)

# for i, sample_output in enumerate(output['sequences']):
#     print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
