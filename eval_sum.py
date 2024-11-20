import pickle as pkl
from rouge_score import rouge_scorer
from alive_progress import alive_bar
from pandas import DataFrame

sumData : DataFrame = pkl.load(open("billsum_inference_results.pkl", "rb"))

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
rouge1 = 0
rouge2 = 0
rougeL = 0
run_time = 0
memory_footprint = 0

for index, row in sumData.iterrows():
	ground_truth = row["ground_truth"]
	predicted_summary = row["predicted_summary"]

	scores = scorer.score(ground_truth, predicted_summary)
	rouge1 += scores["rouge1"].fmeasure
	rouge2 += scores["rouge2"].fmeasure
	rougeL += scores["rougeL"].fmeasure

	run_time += row["run_time"]
	memory_footprint += row["memory_footprint"]

rouge1 /= len(sumData)
rouge2 /= len(sumData)
rougeL /= len(sumData)

run_time /= len(sumData)
memory_footprint /= len(sumData)

print(f"Rouge1: {rouge1}")
print(f"Rouge2: {rouge2}")
print(f"RougeL: {rougeL}")

print(f"Run Time: {run_time}")
print(f"Memory Footprint: {memory_footprint}")