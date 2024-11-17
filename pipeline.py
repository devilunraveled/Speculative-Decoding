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

    def __call__(self, context: str, question: str, maxLen: int = 100) -> dict:
        """
        Run inference on the given context and question.
        """
        inputText = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        # Tokenize the input text.
        inputs = self.model.tokenizer(inputText, return_tensors="pt", max_length=1024, truncation=True).to('cuda')

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
            "context": context,
            "question": question,
            "predicted_answer": outputText,
            "information": information,
            "run_time": runTime,
            "memory_footprint": information.memory_footprint,
        }

if __name__ == '__main__':
    # Load the SQuAD dataset.
    dataset = load_dataset("squad", split="validation")

    dataset = dataset.select(range(3000))

    # Define the two models.
    draftModel = HuggingFaceModelWrapper('openai-community/gpt2').to('cuda')
    mainModel = HuggingFaceModelWrapper('openai-community/gpt2-large').to('cuda')

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
    with alive_bar(len(dataset), length=50, title="SQuAD Inference") as bar:
        for i, data in enumerate(dataset):
            context = data["context"]
            question = data["question"]
            ground_truth = data["answers"]["text"][0] if len(data["answers"]["text"]) > 0 else ""

            # Run inference.
            output = pipeline(context=context, question=question, maxLen=50)

            # Add ground truth for evaluation later.
            output["ground_truth"] = ground_truth

            # Append the result.
            results.append(output)

            # Print progress every 100 samples.
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} samples")

            bar()

    # Convert the results to a DataFrame.
    df = pd.DataFrame(results)

    # Save the DataFrame to a file.
    df.to_pickle("squad_inference_results.pkl")
    df.to_csv("squad_inference_results.csv", index=False)

    print("Inference completed and results saved.")
