from src.decoder import Decoder, SpeculativeDecoder, SimpleDecoder
from src.model import HuggingFaceModelWrapper
from src.utils import getATokenFromTopK
from datasets import load_dataset
import torch
import pandas as pd
import time
from alive_progress import alive_bar

class Pipeline:
    def __init__(self, decoder: Decoder, model: HuggingFaceModelWrapper):
        self.decoder = decoder
        self.model = model

    def __call__(self, title: str, text: str, maxLen: int = 100) -> dict:
        """
        Run inference on the given context and question.
        """
        inputText = f"Title: {title}\nText: {text}\nSummary:"
        
        # Tokenize the input text.
        inputs = self.model.tokenizer(inputText, return_tensors="pt", max_length=512, truncation=True).to('cuda')

        # Generate the output text.
        startTime = time.time()
        generated, _, information = self.decoder.step(inputSeq=inputs.input_ids, numTokens=maxLen)
        runTime = time.time() - startTime

        information.memory_footprint = torch.cuda.memory_allocated() / 1024 / 1024
        information.running_time = runTime

        # Decode the generated text.
        outputText = self.model.tokenizer.decode(generated, skip_special_tokens=True)

        # Return the output along with metadata.
        return {
            "title": title,
            "text": text,
            "predicted_summary": outputText,
            "information": information,
            "run_time": runTime,
            "memory_footprint": information.memory_footprint,
        }

if __name__ == '__main__':
    dataset = load_dataset("billsum", split="test")

    dataset = dataset.select(range(3000))

    # Define the two models.
    draftModel = HuggingFaceModelWrapper('openai-community/gpt2').to("cuda")
    mainModel = HuggingFaceModelWrapper('openai-community/gpt2-large').to("cuda")

    # Build the decoder for the draft model.
    draftModelDecoder = SimpleDecoder(model=draftModel, config={})

    # Initialize the Speculative Decoder.
    speculativeDecoder = SpeculativeDecoder(
        model=mainModel,
        k=5,
        draftModelDecoder=draftModelDecoder,
        samplingScheme=getATokenFromTopK
    )

    # Create the pipeline.
    pipeline = Pipeline(decoder=speculativeDecoder, model=mainModel)

    # Prepare a list to store the results.
    results = []

    # Iterate over the SQuAD dataset for inference.
    try :
        with alive_bar(len(dataset), length=50, title="Billsum Inference") as bar:
            for i, data in enumerate(dataset):
                title = data["title"]
                text = data["text"]
                ground_truth = data["summary"]

                # Run inference.
                output = pipeline(title=title, text=text, maxLen=100)

                # Add ground truth for evaluation later.
                output["ground_truth"] = ground_truth

                print(output["predicted_summary"])

                # Append the result.
                results.append(output)

                # Print progress every 100 samples.
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} samples")

                bar()
    finally:
        # Convert the results to a DataFrame.
        df = pd.DataFrame(results)

        # Save the DataFrame to a file.
        df.to_pickle("billsum_inference_results.pkl")
        df.to_csv("billsum_inference_results.csv", index=False)

        print("Inference completed and results saved.")
