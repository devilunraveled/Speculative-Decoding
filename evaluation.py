import pickle as pkl
import evaluate
import pandas as pd
import pprint

#### SQUAD Evaluation.
def squadEvaluation(filePath, goldAnswersFile ):
    # Load the dataframe from the pickle file.
    data = pd.read_csv(filePath)
    
    totalTimeTaken = sum(data['run_time'])/len(data['run_time'])
    peakMemoryUsage = max(data['memory_footprint'])
    # peakGPUUtilization = max( info.max_util for info in data['information'] )

    with open(goldAnswersFile, 'rb') as f:
        goldAnswers = pkl.load(f)
    
    metric = evaluate.load("squad")
    
    try :
        modelResponses = data['predicted_answer'].tolist()
    except KeyError:
        modelResponses = data['predicted_summary'].tolist()
    predictions = []
    


    for i in range(len(modelResponses)):
        predictions.append(
            {
                'id' : goldAnswers[i]['id'],
                'prediction_text' : modelResponses[i]
            }
        )

    # Assuming 'predictions' is a list of model outputs and 'references' is a list of gold answers
    results = metric.compute(predictions=predictions, references=goldAnswers)
    assert results is not None
    results['run_time_per_sample'] = f"{totalTimeTaken:.2f} s"
    results['peak_memory_footprint'] = f"{peakMemoryUsage:.2f} MB"
    # results['peak_gpu_utilization'] = f"{peakGPUUtilization:.2f}"

    pprint.pprint(results)
if __name__ == '__main__':
    import sys
    fileName = sys.argv[1]
    datasetName = fileName.split('_')[0]
    filePath = f"./results/{datasetName}/{fileName}.csv"
    goldAnswersFile = f"./results/{datasetName}/{datasetName}_gold.pkl"
    squadEvaluation(filePath, goldAnswersFile)
