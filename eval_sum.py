import pickle as pkl
from rouge_score import rouge_scorer
from alive_progress import alive_bar
import pandas as pd
from pandas import DataFrame
import sys

decodingType = sys.argv[1]
modelType = sys.argv[2]

fileName = f"results/billsum/billsum_{decodingType}{'_xl' if modelType == 'xl' else ''}.pkl"
sumData : DataFrame = pd.read_pickle(fileName)
print(f"Loading from {fileName}")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge1 = 0
rouge2 = 0
rougeL = 0
run_time = 0
memory_footprint = 0
max_util = 0

for index, row in sumData.iterrows():
	ground_truth = row["ground_truth"]
	predicted_summary = row["predicted_summary"]

	scores = scorer.score(ground_truth, predicted_summary)
	rouge1 += scores["rouge1"].fmeasure
	rouge2 += scores["rouge2"].fmeasure
	rougeL += scores["rougeL"].fmeasure

	run_time += row["run_time"]
	memory_footprint += row["memory_footprint"]
	max_util += row["information"].max_util

rouge1 /= len(sumData)
rouge2 /= len(sumData)
rougeL /= len(sumData)

run_time /= len(sumData)
memory_footprint /= len(sumData)
max_util /= len(sumData)

print(f"Rouge1: {rouge1}")
print(f"Rouge2: {rouge2}")
print(f"RougeL: {rougeL}")

print(f"Run Time: {run_time}")
print(f"Memory Footprint: {memory_footprint}")
print(f"Max Utilization: {max_util}")