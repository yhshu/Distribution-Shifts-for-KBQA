# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import evaluate


def bleu(predictions, references):
    """
    https://github.com/huggingface/evaluate/tree/main/metrics/bleu
    :param predictions:
    :param references:
    :return:
    """
    bleu = evaluate.load("bleu")  # default BLEU-4
    results = bleu.compute(predictions=predictions, references=references)
    return results


def average_rouge_l(predictions, references):
    """
    https://github.com/huggingface/evaluate/tree/main/metrics/rouge
    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    :param predictions:
    :param references:
    :return:
    """
    rouge = evaluate.load("rouge")  # default ROUGE-L
    results = rouge.compute(predictions=predictions, references=references, use_aggregator=True)
    return results['rougeL']
