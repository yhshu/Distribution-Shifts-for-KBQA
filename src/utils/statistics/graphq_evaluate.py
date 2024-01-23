# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# !/usr/bin/python

import json
import os.path
import sys

import numpy as np

res_file = sys.argv[1]
fb_version = sys.argv[2]  # '13', '15'

graphq_test_15_qid_set = None

graphq_test_15_path = '../dataset/GraphQuestions/graphquestions_v1_fb15_test_091420.json'
if os.path.isfile(graphq_test_15_path):
    graphq_test_15_qid_set = set()
    with open(graphq_test_15_path) as f:
        data = json.load(f)
        for item in data:
            graphq_test_15_qid_set.add(item['qid'])
else:
    print('Warning: graphq_15_test not found')


def computeF1(goldList, predictedList):
    '''
    return a tuple with recall, precision, and f1 for one example
    credit of this function goes to Xuchen Yao
    '''

    '''Assume all questions have at least one answer'''
    if len(goldList) == 0:
        raise Exception('gold list may not be empty')
    '''If we return an empty list recall is zero and precision is one'''
    if len(predictedList) == 0:
        return (0, 1, 0)
    '''It is guaranteed now that both lists are not empty'''

    precision = 0
    for entity in predictedList:
        if entity in goldList:
            precision += 1
    precision = float(precision) / len(predictedList)

    recall = 0
    for entity in goldList:
        if entity in predictedList:
            recall += 1
    recall = float(recall) / len(goldList)

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * recall * precision / (precision + recall)
    return (recall, precision, f1)


m = {}
m['qid'] = 0
m['time'] = 1
m['answers'] = 2
m['predictions'] = 3
m['structure'] = 4
m['function'] = 5
m['answer_cardinality'] = 6
m['commonness'] = 7
m['precision'] = 8
m['recall'] = 9
m['f1'] = 10

# Go over all lines, record information, and compute recall, precision and F1
res = []
with open(res_file) as f:
    if res_file.endswith('.json'):
        data = json.load(f)
        for qid in data:
            try:
                item = data[qid]
                time = item['time']
                answers = item['answers']
                predictions = item['predictions']
                structure = item['structure']
                function = item['function']
                answer_cardinality = item['answer_cardinality']
                commonness = item['commonness']
            except Exception as e:
                print(e)
                continue
            recall, precision, f1 = computeF1(answers, predictions)
            res.append([int(qid), time, answers, predictions, structure,
                        function, answer_cardinality, commonness,
                        precision, recall, f1])
    else:
        for line in f:
            if len(line) == 0 or line[0] == '#':
                continue

            tokens = line.replace('\'', '\"').replace(', \"', ',\"').split('\t')
            try:
                qid = int(tokens[m['qid']])
                time = float(tokens[m['time']])
                answers = json.loads(tokens[m['answers']])
                predictions = json.loads(tokens[m['predictions']])
                structure = int(tokens[m['structure']].split(',')[1])
                function = tokens[m['function']]
                answer_cardinality = int(tokens[m['answer_cardinality']])
                commonness = float(tokens[m['commonness']])
            except Exception as e:
                print(e)
                continue
            recall, precision, f1 = computeF1(answers, predictions)
            res.append([qid, time, answers, predictions, structure,
                        function, answer_cardinality, commonness,
                        precision, recall, f1])


def print_result(options):
    '''
    print the average results over the subset of the questions meeting
    a list of conditions (options).

    An option is a 3-tuple (field, value, operator), e.g.,
    (m['structure'], 1, '==') select questions whose number of edges equals 1.

    AND multiple options.
    '''
    averageRecall = 0
    averagePrecision = 0
    averageF1 = 0
    count = 0
    time = 0
    f1 = []
    for e in res:
        if fb_version == '15' and graphq_test_15_qid_set is not None and int(e[m['qid']]) not in graphq_test_15_qid_set:
            continue
        flag = True
        if not len(options) == 0:
            for op in options:
                field = op[0]
                value = op[1]
                operator = op[2]
                if (operator == '==' and e[field] != value) \
                        or (operator == '!=' and e[field] == value) \
                        or (operator == '>' and e[field] <= value) \
                        or (operator == '>=' and e[field] < value) \
                        or (operator == '<' and e[field] >= value) \
                        or (operator == '<=' and e[field] > value):
                    flag = False
        if flag:  # len(options) == 0
            averageRecall += e[m['recall']]
            averagePrecision += e[m['precision']]
            averageF1 += e[m['f1']]
            f1.append(e[m['f1']])
            time += e[m['time']]
            count += 1
    if count != 0:
        print('\t'.join(
            ['count:' + str(count), 'p:' + str(float(averagePrecision) / count), 'r:' + str(float(averageRecall) / count),
             'f1 avg:' + str(float(averageF1) / count), 'f1 std:' + str(np.std(f1)), 'time:' + str(float(time) / count)]))
    else:
        print('\t'.join(['count: 0.0', 'p: 0.0', 'r: 0.0', 'f1 avg: 0.0', 'f1 std: 0.0', 'time: 0.0']))


def print_result_individual(options, fields2print):
    '''
    print individual question results
    '''

    for e in res:
        flag = True
        if not len(options) == 0:
            # AND multiple options
            for op in options:
                field = op[0]
                value = op[1]
                operator = op[2]
                if (operator == '==' and e[field] != value) \
                        or (operator == '!=' and e[field] == value) \
                        or (operator == '>' and e[field] <= value) \
                        or (operator == '>=' and e[field] < value) \
                        or (operator == '<' and e[field] >= value) \
                        or (operator == '<=' and e[field] > value):
                    flag = False
        if flag and not len(fields2print) == 0:
            s = ''
            for field in fields2print:
                s += str(e[m[field]]) + '\t'
            print(s)
        # ------------------overall-------------------


print('overall performance')
options = []
print_result(options)
# ------------------structure-------------------
print('nEdge = 1')
options = []
options.append([m['structure'], "2,1", '=='])
print_result(options)
print('nEdge = 2')
options = []
options.append([m['structure'], "3,2", '=='])
print_result(options)
print('nEdge = 3')
options = []
options.append([m['structure'], "4,3", '=='])
print_result(options)
# ------------------function-------------------
print('function = none')
options = []
options.append([m['function'], 'none', '=='])
print_result(options)
print('function = count')
options = []
options.append([m['function'], 'count', '=='])
print_result(options)
print('function = superlative')
options = []
options.append([m['function'], 'superlative', '=='])
print_result(options)
print('function = comparative')
options = []
options.append([m['function'], 'comparative', '=='])
print_result(options)
# ------------------answer cardinality-------------------
print('answer_card = 1')
options = []
options.append([m['answer_cardinality'], 1, '=='])
print_result(options)
print('answer_card > 1')
options = []
options.append([m['answer_cardinality'], 1, '>'])
print_result(options)
# ------------------commonness-------------------
print('-40 <= commonness < -30')
options = []
options.append([m['commonness'], -40, '>='])
options.append([m['commonness'], -30, '<'])
print_result(options)
print('-30 <= commonness < -20')
options = []
options.append([m['commonness'], -30, '>='])
options.append([m['commonness'], -20, '<'])
print_result(options)
print('-20 <= commonness < -10')
options = []
options.append([m['commonness'], -20, '>='])
options.append([m['commonness'], -10, '<'])
print_result(options)
print('-10 <= commonness < 0')
options = []
options.append([m['commonness'], -10, '>='])
options.append([m['commonness'], 0, '<'])
print_result(options)


# ------------------paraphrase-------------------
def analyze_paraphrasing():
    pmap = {}  # tid -> f1 list
    std = 0.0

    for e in res:  # for each entry
        if fb_version == '15' and graphq_test_15_qid_set is not None and int(e[m['qid']]) not in graphq_test_15_qid_set:
            continue

        qid = e[m['qid']]
        f1 = e[m['f1']]
        tid = qid // 1000000  # template id
        if tid in pmap:
            pmap.get(tid).append(f1)
        else:
            pmap[tid] = [f1]

    n_max = 0  # the max length of f1 list
    for key in pmap.keys():
        pmap.get(key).sort(reverse=True)
        if len(pmap.get(key)) > n_max:
            n_max = len(pmap.get(key))
        std += np.std(pmap.get(key))
    std = std / len(pmap.keys())

    for i in range(0, n_max):
        f1 = 0
        n = 0
        for key in pmap.keys():
            l = pmap.get(key)  # f1 list
            if len(l) > i:
                f1 += l[i]
                n += 1
        if n != 0:
            f1 /= n
        print('\t'.join([str(i), str(n), str(f1)]))
    print('paraphrase f1 std: ' + str(std))


print('paraphrasing')
analyze_paraphrasing()
