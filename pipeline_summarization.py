from src.decoder import SpeculativeDecoder, SimpleDecoder
from src.model import HuggingFaceModelWrapper
from src.utils import getATokenFromTopK, getMostProbableToken
from pipeline import Pipeline
from datasets import load_dataset
import pandas as pd
from alive_progress import alive_bar

if __name__ == "__main__":
    dataset = load_dataset("billsum", split="test")

    dataset = dataset.select(range(3000))

    # Define the two models.
    # draftModel = HuggingFaceModelWrapper('openai-community/gpt2').to("cuda")
    mainModel = HuggingFaceModelWrapper('openai-community/gpt2-large').to("cuda")

    # Build the decoder for the draft model.
    # draftModelDecoder = SimpleDecoder(model=draftModel, config={})

    # Initialize the Speculative Decoder.
    # speculativeDecoder = SpeculativeDecoder(
    #     model=mainModel,
    #     k=5,
    #     draftModelDecoder=draftModelDecoder,
    #     samplingScheme=getATokenFromTopK
    # )
    simpleDecoder = SimpleDecoder(model=mainModel, samplingScheme=getATokenFromTopK, config={"eosTokenId": mainModel.tokenizer.eos_token_id})

    # Create the pipeline.
    pipeline = Pipeline(decoder=simpleDecoder, model=mainModel)

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
                output = pipeline(title=title, text=text, maxLen=50)

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
        df.to_pickle("billsum_inference_results_simple.pkl")
        df.to_csv("billsum_inference_results_simple.csv", index=False)

        print("Inference completed and results saved.")
