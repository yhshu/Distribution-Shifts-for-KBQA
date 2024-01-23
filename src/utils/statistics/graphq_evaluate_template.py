# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')
import json
from utils.file_util import write_json_file

if __name__ == '__main__':
    result_sample_path = '../dataset/GraphQuestions/results/sempre.res'
    res = {}
    with open(result_sample_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line_split = line.split('\t')
            qid = int(line_split[0])
            item = {'time': 0.0, 'answers': json.loads(line_split[2]), 'predictions': [],
                    'structure': line_split[4], 'function': line_split[5], 'answer_cardinality': int(line_split[6]), 'commonness': float(line_split[7])}
            res[qid] = item

    write_json_file('../dataset/GraphQuestions/results/sample.json', res)
