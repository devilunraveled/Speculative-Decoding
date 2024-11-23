import pandas as pd
from pprint import pprint

def getInformationForSpeculativeDecoder(informationObjects, datasetName : str ):
    totalTime = sum(info.running_time for info in informationObjects)
    timePerSample = totalTime / len(informationObjects)
    
    totalDraftedTokens = sum(info.drafted for info in informationObjects)
    draftRejectedTokens = sum(info.reject for info in informationObjects)
    draftAcceptedTokens = sum(info.accept_direct + info.accept_indirect for info in informationObjects)

    peakVRAM = max(info.memory_footprint for info in informationObjects)
    
    totalGenerated = sum(info.total_generated for info in informationObjects)
    modelGenerated = totalGenerated - draftAcceptedTokens
    
    print(f"Average Length : {sum(info.total_generated for info in informationObjects) / len(informationObjects)}")
    
    if totalDraftedTokens != 0 :
        # Speculative Scheme Somewhere.
        direcAcceptence = sum(info.accept_direct for info in informationObjects) / totalDraftedTokens
        print(f"Direct Acceptence Rate of Model : {direcAcceptence*100:.2f}")
        totalAccepted = draftAcceptedTokens / totalDraftedTokens
        print(f"Total Acceptence Rate of Model : {totalAccepted*100:.2f}")

    try :
        timePerToken = totalTime / totalGenerated
    except ZeroDivisionError:
        timePerToken = 0

        if datasetName == "squad" :
            goldAnswers = pd.read_pickle(f"./results/{datasetName}/{datasetName}_gold.pkl")
            lenGoldAnswers = [max( len(answer) for answer in ground_truth['answers']['text']) for ground_truth in goldAnswers ]

            totalGenerated = sum(lenGoldAnswers)
            modelGenerated = totalGenerated - draftAcceptedTokens
            
            timePerToken = totalTime / totalGenerated
        elif datasetName == "storygen" :
            lenGoldAnswers = [80 for _ in range(len(informationObjects))]

            timePerToken = totalTime / sum(lenGoldAnswers)
        elif datasetName == "billsum" :
            lenGoldAnswers = [100 for _ in range(len(informationObjects))]
            timePerToken = totalTime / sum(lenGoldAnswers)


    try :
        token_per_main_model_pass = totalGenerated / modelGenerated
    except ZeroDivisionError:
        token_per_main_model_pass = 0


    return {
        'time_per_sample' : f"{timePerSample:.4f} s",
        'time_per_token' : f"{timePerToken:.4f} s",
        'total_draft_tokens' : totalDraftedTokens,
        'draft_rejected_tokens' : draftRejectedTokens,
        'draft_accepted_tokens' : draftAcceptedTokens,
        'model_generated_tokens' : modelGenerated,
        'total_generated_tokens' : totalGenerated,
        'token_per_main_model_pass' : f"{token_per_main_model_pass:.4f}",
        'peak_vram' : f"{peakVRAM:.2f} MB",
    }


#### SQUAD Results.
def results(filePath, dataset):
    data = pd.read_pickle(filePath)
    if 'nested' in filePath :
        layer1Decoders = data['information'].tolist()
        print("Layer 1 Decoders : ")
        pprint(getInformationForSpeculativeDecoder(layer1Decoders, dataset))
        layer2Decoders = [info.subDecoder_info for info in data['information'].tolist()]
        print("Layer 2 Decoders :")
        pprint(getInformationForSpeculativeDecoder(layer2Decoders, dataset))
    else:
        # Compute total number of draft tokens
        pprint(getInformationForSpeculativeDecoder(data['information'].tolist(), dataset))

if __name__ == '__main__':
    import sys
    fileName = sys.argv[1]
    datasetName = fileName.split('_')[0]
    filePath = f"./results/{datasetName}/{fileName}.pkl"
    results(filePath, datasetName)
