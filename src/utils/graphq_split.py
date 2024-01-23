# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random

from dataloader.graphq_json_loader import GraphQuestionsJsonLoader
from utils.config import graphq_test_path, graphq_train_path
from utils.file_util import write_json_file

if __name__ == '__main__':
    graphq_train = GraphQuestionsJsonLoader(graphq_train_path)
    graphq_test = GraphQuestionsJsonLoader(graphq_test_path)

    random.seed(429)
    dev_idx = [i for i in range(0, graphq_train.len)]
    random.shuffle(dev_idx)
    dev_idx = dev_idx[:200]

    train_data = []
    dev_data = []
    for idx in range(0, graphq_train.len):
        if idx in dev_idx:
            dev_data.append(graphq_train.data[idx])
        else:
            train_data.append(graphq_train.data[idx])

    write_json_file('../dataset/GraphQuestions/graph_questions_ptrain.json', train_data)
    print('train data size: {}'.format(len(train_data)))
    write_json_file('../dataset/GraphQuestions/graph_questions_pdev.json', dev_data)
    print('dev data size: {}'.format(len(dev_data)))
