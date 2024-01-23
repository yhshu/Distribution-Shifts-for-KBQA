# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')

from tqdm import tqdm

from retriever.freebase_retriever import FreebaseRetriever
from utils.file_util import write_list_file

if __name__ == '__main__':
    retriever = FreebaseRetriever()

    with open('../dataset/GrailQA/ontology/fb_roles', 'r') as f:
        fb_roles = f.readlines()
    classes = set()
    relations = set()

    for line in tqdm(fb_roles):
        line_split = line.strip('\n').split(' ')
        relation_domain = line_split[0]
        relation = line_split[1]
        relation_range = line_split[2]

        classes.add(relation_domain)
        classes.add(relation_range)
        relations.add(relation)

    valid_classes = set()
    for c in tqdm(classes):
        if retriever.judge_class_exists(c):
            valid_classes.add(c)

    valid_relations = set()
    for r in tqdm(relations):
        if retriever.judge_relation_exists(r):
            valid_relations.add(r)

    write_list_file(valid_classes, '../dataset/fb_roles_classes.txt')
    write_list_file(valid_relations, '../dataset/fb_roles_relations.txt')
    print('done')
