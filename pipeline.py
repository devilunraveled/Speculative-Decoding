from src.decoder import SpeculativeDecoder, SimpleDecoder
from src.model import HuggingFaceModelWrapper

class Pipeline:
    def __init__(self, decoder : SpeculativeDecoder):
        self.decoder = decoder

    def __call__(self, inputText : str, maxLen : int = 100) -> str:
        # Tokenize the input text.
        inputs = self.decoder.model.tokenizer(inputText, return_tensors="pt", max_length=512)

        # Generate the output text.
        generated = self.decoder.step(inputSeq = inputs.input_ids, maxLen = maxLen)

        # Decode the generated text.
        outputText = self.decoder.model.tokenizer.decode(generated, skip_special_tokens=True)
        
        return outputText

if __name__ == '__main__' :
    # Define the two models.
    draftModel = HuggingFaceModelWrapper('openai-community/gpt2')
    mainModel = HuggingFaceModelWrapper('openai-community/gpt2-xl')
    
    # Build the decoder for the draft model.
    draftModelDecoder = SimpleDecoder(model = draftModel, config = {})

    # Initialize the Speculative Decoder 
    speculativeDecoder = SpeculativeDecoder(
        model = mainModel,
        k = 5,
        draftModelDecoder = draftModelDecoder
    )

    # Create the pipeline.
    pipeline = Pipeline(decoder = speculativeDecoder)

    # Generate the output text.
    outputText = pipeline(inputText = "Hello", maxLen = 100)

    # Print the output text.
    print(outputText)
