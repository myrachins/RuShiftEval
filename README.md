# Zero-shot Cross-lingual Transfer of a Gloss Language Model for Semantic Change Detection

#### This is code for the 1st system in [RuShiftEval](https://competitions.codalab.org/competitions/28340) competition.

## How to run

### Step 0: Prepare environment
1. Install [python](https://python.org/) 3.7 or later.
1. Clone repo and move to the root directory (i.e., `RuShiftEval`).
1. Run `pip install -r requirements.txt` to install the required packages.
1. To run the code with the default precomputed parameters, download `data` directory from [drive](https://yadi.sk/d/CIU9Hm0tvKPH2g).
1. Place `data` directory in the root of the project (i.e., `RuShiftEval/data`).

### Step 1: Generate predictions file
Run `python base_large_linear.py` to generate our best predictions (*Linear regression on GLM xlmr.large+base distances* from the results table).
Additionally, you can change the default parameters: 
- `--rand-seed` - random seed for the whole program.
- `--train-val-size` - validation part of the train data.
***
- `--rusemshift-1-data` - path to json formatted data *raw_annotations_1* from RuSemShift dataset.
- `--rusemshift-1-gold` - path to json formatted labels *raw_annotations_1* from RuSemShift dataset.
- `--rusemshift-1-base-vectors` - computed vectors from the base encoder for each sample from *raw_annotations_1*.
- `--rusemshift-1-large-vectors` - computed vectors from the large encoder for each sample from *raw_annotations_1*.
***
- `--rusemshift-2-data` - path to json formatted data *raw_annotations_2* from RuSemShift dataset.
- `--rusemshift-2-gold` - path to json formatted labels *raw_annotations_2* from RuSemShift dataset.
- `--rusemshift-2-base-vectors` - computed vectors from the base encoder for each sample from *raw_annotations_2*.
- `--rusemshift-2-large-vectors` - computed vectors from the large encoder for each sample from *raw_annotations_2*.
***
- `--epochs-samples-dir-12` - path to the sampled sentences from pre-Soviet:Soviet epochs pair.
- `--epochs-base-vectors-12` - computed vectors from the base encoder for the sentences from pre-Soviet:Soviet epochs pair.
- `--epochs-large-vectors-12` - computed vectors from the large encoder for the sentences from pre-Soviet:Soviet epochs pair.
***
- `--epochs-samples-dir-13` - path to the sampled sentences from pre-Soviet:post-Soviet epochs pair.
- `--epochs-base-vectors-13` - computed vectors from the base encoder for the sentences from pre-Soviet:post-Soviet epochs pair.
- `--epochs-large-vectors-13` - computed vectors from the large encoder for the sentences from pre-Soviet:post-Soviet epochs pair.
***
- `--epochs-samples-dir-23` - path to the sampled sentences from Soviet:post-Soviet epochs pair.
- `--epochs-base-vectors-23` - computed vectors from the base encoder for the sentences from Soviet:post-Soviet epochs pair.
- `--epochs-large-vectors-23` - computed vectors from the large encoder for the sentences from Soviet:post-Soviet epochs pair.
***
- `--output-predictions-path` - path where to store predictions.

### Step 2: Evaluate predictions
Run `python test_score.py` to calculate test scores for a prediction file.
Additionally, you can change the default parameters: 
- `--predictions-filepath` - path to the generated predictions.
- `--gold-scores-filepath` - path to the gold scores.
 
## Published parts
- 🔲 GLM pre-training code.
- 🔲 Inference code for obtaining contextualized GLM pre-trained embeddings. 
- ✅ Prediction generation code based on the precomputed GLM embeddings. 
- ✅ Evaluation code for the predictions.


## Authors
- Maxim Rachinskiy
- Nikolay Arefyev