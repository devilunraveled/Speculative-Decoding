from src.decoder import Decoder, SpeculativeDecoder, SimpleDecoder
from src.model import HuggingFaceModelWrapper
from src.utils import getATokenFromTopK

class Pipeline:
    def __init__(self, decoder : Decoder, model : HuggingFaceModelWrapper):
        self.decoder = decoder
        self.model = model

    def __call__(self, inputText : str, maxLen : int = 100) -> str:
        # Tokenize the input text.
        inputs = self.model.tokenizer(inputText, return_tensors="pt", max_length=512, truncation=True).to('cuda')

        # Generate the output text.
        generated, _, acceptences = self.decoder.step(inputSeq = inputs.input_ids, numTokens = maxLen)

        # Decode the generated text.
        outputText = self.model.tokenizer.decode(generated, skip_special_tokens=True)
        print(acceptences)
        return outputText

if __name__ == '__main__' :
    # Define the two models.
    draftModel = HuggingFaceModelWrapper('openai-community/gpt2').to('cuda')
    mainModel = HuggingFaceModelWrapper('openai-community/gpt2-large').to('cuda')
    
    import sys

    inputText = sys.argv[1]

    # Build the decoder for the draft model.
    draftModelDecoder = SimpleDecoder(model = draftModel, config = {})

    # Initialize the Speculative Decoder 
    speculativeDecoder = SpeculativeDecoder(
        model = mainModel,
        k = 5,
        draftModelDecoder = draftModelDecoder, 
        samplingScheme = getATokenFromTopK
    )
    
    simpleDecoder = SimpleDecoder(model = mainModel, config = {})

    # Create the pipeline.
    pipeline = Pipeline(decoder = speculativeDecoder, model = mainModel)

    # Generate the output text.
    outputText = pipeline(inputText = inputText, maxLen = 200)

    # Print the output text.
    print(f"Prompt : {inputText}\nOutput : {outputText}")
