from transformers.utils import quantization_config
from src.decoder import Decoder, SpeculativeDecoder, SimpleDecoder, BeamSearchDecoder
from src.model import HuggingFaceModelWrapper
from src.utils import getATokenFromTopK, getMostProbableToken
from datasets import load_dataset
import torch
import pandas as pd
import time
from alive_progress import alive_bar
import pprint

class Pipeline:
    def __init__(self, decoder: Decoder, model: HuggingFaceModelWrapper):
        self.decoder = decoder
        self.model = model

    def __call__(self, prompt : str, maxLen: int, *args, **kwargs) -> dict:
        """
        Run inference on the given context and question.
        """
        # Tokenize the input text.
        inputs = self.model.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to('cuda')

        # Generate the output text.
        startTime = time.time()
        generated, _, information = self.decoder.step(
            inputSeq=inputs.input_ids, 
            numTokens=maxLen,
            *args, **kwargs
        )
        runTime = time.time() - startTime

        information.memory_footprint = torch.cuda.memory_allocated() / 1024 / 1024
        information.running_time = runTime

        # Decode the generated text.
        outputText = self.model.tokenizer.decode(generated, skip_special_tokens=True)

        # Return the output along with metadata.
        return {
            "prompt" : prompt,
            "predicted_output": outputText,
            "information": information,
            "run_time": runTime,
            "memory_footprint": information.memory_footprint,
        }

if __name__ == '__main__':
    import sys
    datasetName = sys.argv[1]
    decodingType = sys.argv[2]

    if datasetName == 'storygen' :
        datapointLimit = 1000
        with open("data/valid.wp_source", "r") as f:
            prompts = f.readlines()

        dataset = [{"prompt": prompt} for prompt in prompts]
        
        with open("data/valid.wp_target", "r") as f:
            stories = f.readlines()

        for i, story in enumerate(stories):
            dataset[i]["story"] = story

        dataset = dataset[:datapointLimit]

    else :
        dataset = load_dataset(datasetName, split="validation" if datasetName == 'squad' else 'test')

        dataset = dataset.select(range(5000 if datasetName == 'squad' else 1000))

    # Define the two models.
    if decodingType == 'speculative' :
        draftModel = HuggingFaceModelWrapper('openai-community/gpt2-large').to('cuda')
        draftModelDecoder = SimpleDecoder(
            model=draftModel, 
            config={},
            samplingScheme=getATokenFromTopK
        )
        print(f"Loaded Draft Model: {draftModel.model.name_or_path}")
    elif decodingType == 'nested_speculative' :
        draftModel1 = HuggingFaceModelWrapper('openai-community/gpt2').to('cuda')
        draftModel1Decoder = SimpleDecoder(
            model=draftModel1, 
            config={},
            samplingScheme=getATokenFromTopK
        )
        print(f"Loaded Draft Model Base : {draftModel1.model.name_or_path}")
        draftModel2 = HuggingFaceModelWrapper('openai-community/gpt2-large').to('cuda')
        draftModelDecoder = SpeculativeDecoder(
            model=draftModel2,
            gamma=5,
            draftModelDecoder=draftModel1Decoder,
            samplingScheme=getATokenFromTopK,
            k = 10
        )
        print(f"Loaded Draft Model Middle : {draftModel2.model.name_or_path}")
        
    
    mainModel = HuggingFaceModelWrapper('openai-community/gpt2-xl').to('cuda')
    print(f"Loaded Main Model: {mainModel.model.name_or_path}")

    # Initialize the Speculative Decoder.
    if 'speculative' in decodingType :
        mainModelDecoder = SpeculativeDecoder(
            model=mainModel,
            gamma=5,
            draftModelDecoder=draftModelDecoder,
            samplingScheme=getATokenFromTopK,
            k = 10
        )
    elif decodingType == 'beam' :
            mainModelDecoder = BeamSearchDecoder(
                model=mainModel,
                beamSize=3,
                samplingScheme=getATokenFromTopK,
                config={},
            )
    else :
        mainModelDecoder = SimpleDecoder(
            model=mainModel, 
            config={},
            samplingScheme=getATokenFromTopK if decodingType == 'topk' else getMostProbableToken
        )
    
    pipeline = Pipeline(decoder=mainModelDecoder, model=mainModel)

    # Prepare a list to store the results.
    results = []

    # Iterate over the SQuAD dataset for inference.
    try :
        with alive_bar(len(dataset), length=10, title=f"{datasetName}-{decodingType}", force_tty=True) as bar:
            for i, data in enumerate(dataset):
                maxLen = 100
                inputText = "Hello"
                ground_truth = "NO_VALID_ANSWER"

                if datasetName == 'billsum' :
                    title = data["title"]
                    text = data["text"]
                    inputText = f"Title: {title}\nText: {text}\nSummary:"
                    
                    ground_truth = data["summary"]
                    maxLen = 80
                elif datasetName == 'squad' :
                    context = data["context"]
                    question = data["question"]
                    inputText = f"Given the \nContext: {context}\n Answer this Question: {question}\n"
                    
                    ground_truth = {
                        'id'        : data['id'],
                        'answers'    : data['answers']
                    }
                    maxLen=max((len(answer)) for answer in ground_truth['answers']['text'])
                elif datasetName == 'storygen' :
                    inputText = data["prompt"]
                    ground_truth = data["story"]
                    maxLen = 100

                output = pipeline(prompt = inputText, maxLen=maxLen)


                # Add ground truth for evaluation later.
                output["ground_truth"] = ground_truth

                # Append the result.
                results.append(output)
                
                bar()
    finally:
        # Convert the results to a DataFrame.
        df = pd.DataFrame(results)

        # Save the DataFrame to a file.
        df.to_pickle(f"{datasetName}_{decodingType}_large_as_draft.pkl")
        df.to_csv(f"{datasetName}_{decodingType}_large_as_draft.csv", index=False)
        
        print("Inference completed and results saved.")
