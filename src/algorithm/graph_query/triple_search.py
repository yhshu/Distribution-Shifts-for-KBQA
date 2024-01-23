# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('dataloader')

import argparse
from transformers.hf_argparser import string_to_bool
from dataloader.simple_questions_balance_data_loader import SimpleQuestionsBalanceDataloader
from tqdm import tqdm
from retriever.freebase_retriever import FreebaseRetriever
from utils.file_util import read_set_file, write_json_file
from utils.uri_util import remove_ns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_seen', type=str, default='False')
    parser.add_argument('--num_head', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='../dataset/question_generation')
    args = parser.parse_args()

    head_limit = args.num_head
    skip_seen = string_to_bool(args.skip_seen)

    with open('../dataset/GrailQA/ontology/fb_roles', 'r') as f:
        fb_roles = f.readlines()
    relation_to_entities = {}
    for line in fb_roles:
        line_split = line.strip('\n').split(' ')  # relation domain, relation, relation range
        relation_domain = line_split[0]
        relation = line_split[1]
        relation_range = line_split[2]
        relation_to_entities[relation] = (relation_domain, relation_range)

    fb_entity_classes = read_set_file('../dataset/freebase_entity_classes.txt')
    sqb_train = SimpleQuestionsBalanceDataloader('../dataset/SQB/fold-0.train.pickle')
    sqb_train_relations = sqb_train.get_golden_relations()
    retriever = FreebaseRetriever()

    res = []
    total_count = 0
    for r in tqdm(relation_to_entities):  # for each relation
        if skip_seen is True and r in sqb_train_relations:
            continue
        triple_count = 0
        r_domain, r_range = relation_to_entities[r]
        if r_domain not in fb_entity_classes or r_range not in fb_entity_classes:  # head or tail is not entity
            continue
        head_entities = retriever.entity_by_relation(r, head=True, tail=False, no_literal=True)
        head_count = 0
        for head in head_entities:
            tail_entities = retriever.neighbor_by_mid_and_relation(head, r, forward=True, backward=False)
            if len(tail_entities) > 10:  # the setting of SimpleQuestions: the tail entities are too many
                continue
            found = False
            for tail in tail_entities:
                if tail['type'] != 'uri':  # tail is not entity
                    continue
                head_name = retriever.rdf_label_by_mid(head, only_one=True)
                tail_name = retriever.rdf_label_by_mid(tail['value'], only_one=True)
                if len(head_name) and len(tail_name):  # the head and the tail are entities
                    triple_count += 1
                    res.append((remove_ns(head), remove_ns(r), remove_ns(tail), head_name, tail_name))
                    found = True
                    break
            if found is True:
                head_count += 1
            if head_count >= head_limit:  # the max number of head entities for this relation
                break
        total_count += triple_count
        print(r, '#triple:', triple_count, '#total:', total_count)

    retriever.save_cache()
    write_json_file(args.output_dir + '/triple_fb.json', res)
    print('done')
