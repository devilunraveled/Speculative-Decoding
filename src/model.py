from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HuggingFaceModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model.eval()
	
    def infer(self, inputSeq, lastK = 1):
        """
        :param inputSeq: a (batched) sequence of tokens.
        :return: a (batched) distribution over the vocabulary.
        """
        with torch.no_grad():
            return torch.softmax(self.model(inputSeq, output_hidden_states=False).logits[0, -lastK:, :], dim=-1)
