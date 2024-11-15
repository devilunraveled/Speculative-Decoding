import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.decoder import SimpleDecoder, BeamSearchDecoder
from src.model import Inferencer, HuggingFaceModelWrapper

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", load_in_4bit=True)
modelWrapper = HuggingFaceModelWrapper(model)
# # set model to evaluation mode
# model.eval()

# text = "Who are you?"
# outputTokens = []
# with torch.no_grad():
# 	input_ids = tokenizer(text, return_tensors="pt").input_ids
# 	for i in range(100):
# 		output = model(input_ids, output_hidden_states=False)
# 		outputTokens.append(output.logits[:, -1, :].argmax())
# 		input_ids = torch.cat([input_ids, torch.tensor([[outputTokens[-1]]])], dim=-1)

# print(tokenizer.decode(input_ids[0]))
config = {
	"eosTokenId": tokenizer.eos_token_id
}

greedyDecoder = SimpleDecoder(modelWrapper, config)
beamDecoder = BeamSearchDecoder(modelWrapper, 3, config)

prompt = "what is apple ?"

greedyInferencer = Inferencer(tokenizer, greedyDecoder)
print("Greedy Inferencer: ", greedyInferencer.generate(prompt, maxLen = 50))

beamInferencer = Inferencer(tokenizer, beamDecoder)
print("Beam Inferencer: ", beamInferencer.generate(prompt, maxLen = 50))