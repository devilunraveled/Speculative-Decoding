# Speculative-Decoding
The implementation and analysis of Speculative Decoding, implemented as part of Project work for the course CS7.501 Advanced Natural Language Processing

## Running the code for Inference.

You can run the code for inference by simply running the following command :

`python pipeline.py [datasetName] [decodingScheme]`

`datasetNames = ['squad', 'billsum', 'storygen'], decodingScheme = ['beam', 'greedy', 'topk', 'speculative', 'nested_specuative']`

## Running the code for Evaluation.

Evaluations can be run by using the separate evaluation scripts available in the code along with the dataset and scheme as described above.
