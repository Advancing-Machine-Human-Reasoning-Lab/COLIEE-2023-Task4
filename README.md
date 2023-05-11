# AMHR Lab 2023 COLIEE Competition Approach
Code and data for reproducing our results for the COLIEE 2023 Competition, Task 4.

## Installation

All code is based on Python 3.x, we recommend using version 3.9 or higher. Most dependencies can be installed using the ``requirements.txt``. Note that ``BERTScore`` and ``BleuRT`` have special installation instructions that cannot be handled with just pip, please see the project's respective pages for instructions:

[BERTScore: Evaluating Text Generation with BERT](https://github.com/Tiiiger/bert_score)

[BLEURT: a Transfer Learning-Based Metric for Natural Language Generation](https://github.com/google-research/bleurt)

## Summary of Files

1. ``prompt_tuning.py``: All code for prompt-tuning Huggingface and OpenAI models.
2. ``dt.py``: Implementation of ensemble prompting approach.
3. ``master_df.tsv``: Used to train ensemble models. For convenience, we merged all our results from the non-ensemble models into a single file for training the meta classifier. 
4. ``similarity.py``: Implementation of shot selection metrics.
5. ``xml_processing.ipynb``: Converts the raw COLIEE XML data into pandas dataframe for easier processing.

## Note on Training Data

All our scripts assume the training data has already been cleaned and stored as tsv files. You must obtain the COLIEE Task 4 training data yourself from the competition organizers, we do not have permission to share this. Once you have done that and run it through ``xml_processing``, make sure to also change file paths in the other scripts to point to where you stored the cleaned splits.

## Citation

Information about how to cite our work will be released after the conference.