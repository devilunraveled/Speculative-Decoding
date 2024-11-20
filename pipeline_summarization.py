from src.decoder import SpeculativeDecoder, SimpleDecoder
from src.model import HuggingFaceModelWrapper
from src.utils import getATokenFromTopK, getMostProbableToken
from pipeline import Pipeline
from datasets import load_dataset
import pandas as pd
from alive_progress import alive_bar
import sys

if __name__ == "__main__":
    decodingType = sys.argv[1]

    dataset = load_dataset("billsum", split="test")

    dataset = dataset.select(range(1000))

    # Define the two models.
    # draftModel = HuggingFaceModelWrapper('openai-community/gpt2').to("cuda")
    mainModel = HuggingFaceModelWrapper('openai-community/gpt2-large').to("cuda")

    decoder = None
    if decodingType == "greedy" :
        simpleDecoder = SimpleDecoder(model=mainModel, samplingScheme=getMostProbableToken, config={"eosTokenId": mainModel.tokenizer.eos_token_id})
        decoder = simpleDecoder
    elif decodingType == "topk" :
        simpleDecoder = SimpleDecoder(model=mainModel, samplingScheme=getATokenFromTopK, config={"eosTokenId": mainModel.tokenizer.eos_token_id})
        decoder = simpleDecoder
    elif decodingType == "speculative" :
        draftModel =  HuggingFaceModelWrapper('openai-community/gpt2').to("cuda")
        draftModelDecoder = SimpleDecoder(model=draftModel, config={})
        speculativeDecoder = SpeculativeDecoder(
            model=mainModel,
            k=5,
            draftModelDecoder=draftModelDecoder,
            samplingScheme=getATokenFromTopK
        )
        decoder = speculativeDecoder

    # Create the pipeline.
    pipeline = Pipeline(decoder=decoder, model=mainModel)

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
        df.to_pickle(f"billsum_inference_results_{decodingType}.pkl")
        df.to_csv(f"billsum_inference_results_{decodingType}.csv", index=False)

        print("Inference completed and results saved.")
