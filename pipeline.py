from transformers.utils import quantization_config
from src.decoder import Decoder, SpeculativeDecoder, SimpleDecoder
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

    def __call__(self, context: str, question: str, maxLen: int = 100) -> dict:
        """
        Run inference on the given context and question.
        """
        inputText = f"Given the \nContext: {context}\n Answer this Question: {question}\n"
        
        # Tokenize the input text.
        inputs = self.model.tokenizer(inputText, return_tensors="pt", max_length=512, truncation=True).to('cuda')

        # Generate the output text.
        startTime = time.time()
        generated, _, information = self.decoder.step(
            inputSeq=inputs.input_ids, 
            numTokens=maxLen,
            k = 10,
        )
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
    import sys

    decodingType = sys.argv[1]
    
    datasetName = 'squad'
    # Load the SQuAD dataset.
    dataset = load_dataset(datasetName, split="validation")
    dataset = dataset.select(range(5000))

    # Define the two models.
    if decodingType == 'speculative' :
        draftModel = HuggingFaceModelWrapper('openai-community/gpt2').to('cuda')
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
    else :
        mainModelDecoder = SimpleDecoder(
            model=mainModel, 
            config={},
            samplingScheme=getATokenFromTopK if decodingType == 'topk' else getMostProbableToken
        )
    # Create the pipeline.
    pipeline = Pipeline(decoder=mainModelDecoder, model=mainModel)

    # Prepare a list to store the results.
    results = []

    # Iterate over the SQuAD dataset for inference.
    try :
        with alive_bar(len(dataset), length=20, title="SQuAD Inference", force_tty=True) as bar:
            for i, data in enumerate(dataset):
                context = data["context"]
                question = data["question"]
                
                ground_truth = {
                    'id'        : data['id'],
                    'answers'    : data['answers']
                }

                # Run inference.
                output = pipeline(context=context, question=question, maxLen=max((len(answer)) for answer in ground_truth['answers']['text']))
                # Add ground truth for evaluation later.
                output["ground_truth"] = ground_truth
                
                # Append the result.
                results.append(output)
                
                bar()
    finally:
        # Convert the results to a DataFrame.
        df = pd.DataFrame(results)

        # Save the DataFrame to a file.
        df.to_pickle(f"{datasetName}_{decodingType}.pkl")
        df.to_csv(f"{datasetName}_{decodingType}.csv", index=False)
        
        # import pickle
        # # # Save the Gold 
        # goldAnswers = [result['ground_truth'] for result in results]

        # print(goldAnswers[0])
        # with open(f"{datasetName}_gold_temp.pkl", 'wb') as f:
        #     pickle.dump(goldAnswers, f)

        print("Inference completed and results saved.")
