# ITEM
## Download the Datsets
Download TREC-DL test data from [TREC-DL19 and TREC-DL20](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html);
WebAP from [WebAP](https://ciir.cs.umass.edu/downloads/WebAP/index.html); 
NQ from [NQ](https://ai.google.com/research/NaturalQuestions/download)

## Get the retrieval data
BM25 for the TREC and WebAP datasets, RocketQAv2 for the NQ dataset.

## Create the candidate passage list
We have also provided the final dataset in "data/trec.json",  "data/webap.json", and [NQ](https://drive.google.com/file/d/1pAK6CYbuN7qrXg60A_TkN6_sExg6IREg/view?usp=sharing).

## Download the LLMs
Download the LLMs from [huggingface](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 

## ITEM via LLMs
Taking the testing of Mistral-7B as an example, we demonstrated the use of four methods: ITEM-As, ITEM-ARs, and single-shot-utility-judgments
```
python mistral/item-As-ImpA.py

```


