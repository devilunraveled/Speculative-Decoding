import pandas as pd
from pprint import pprint

def getInformationForSpeculativeDecoder(informationObjects):
    totalTime = sum(info.running_time for info in informationObjects)
    timePerSample = totalTime / len(informationObjects)
    
    totalDraftedTokens = sum(info.drafted for info in informationObjects)
    draftRejectedTokens = sum(info.reject for info in informationObjects)
    draftAcceptedTokens = sum(info.accept_direct + info.accept_indirect for info in informationObjects)

    totalGenerated = sum(info.total_generated for info in informationObjects)
    modelGenerated = totalGenerated - draftAcceptedTokens
    
    timePerToken = totalTime / totalGenerated
    
    return {
        'time_per_sample' : f"{timePerSample:.4f} s",
        'time_per_token' : f"{timePerToken:.4f} s",
        'total_draft_tokens' : totalDraftedTokens,
        'draft_rejected_tokens' : draftRejectedTokens,
        'draft_accepted_tokens' : draftAcceptedTokens,
        'model_generated_tokens' : modelGenerated,
        'total_generated_tokens' : totalGenerated,
        'token_per_main_model_pass' : f"{totalGenerated/modelGenerated :.4f}",
    }


#### SQUAD Results.
def results(filePath):
    data = pd.read_pickle(filePath)
    if 'nested' in filePath :
        layer1Decoders = data['information'].tolist()
        pprint(getInformationForSpeculativeDecoder(layer1Decoders))
        layer2Decoders = [info.subDecoder_info for info in data['information'].tolist()]
        pprint(getInformationForSpeculativeDecoder(layer2Decoders))
    elif 'speculative' in filePath:
        # Compute total number of draft tokens
        pprint(getInformationForSpeculativeDecoder(data['information'].tolist()))

if __name__ == '__main__':
    import sys
    fileName = sys.argv[1]
    datasetName = fileName.split('_')[0]
    filePath = f"./results/{datasetName}/{fileName}.pkl"
    results(filePath)
