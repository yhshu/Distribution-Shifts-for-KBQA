# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from py_stringmatching import PartialRatio

from dataloader.json_loader import JsonLoader
from rng.framework.executor.logic_form_util import get_lisp_from_graph_query
from utils.config import graphq_train_path


class GraphQuestionsJsonLoader(JsonLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
        with open(file_path, 'r', encoding='UTF-8') as f:
            self.data = json.load(f)
            self.len = len(self.data)

    def get_question_by_idx(self, idx):
        return self.data[idx]['question']

    def get_question_id_by_idx(self, idx):
        return self.data[idx]['qid']

    def get_graph_query(self, idx):
        return self.data[idx]['graph_query']

    def get_sparql_by_idx(self, idx):
        return self.data[idx]['sparql_query']

    def get_ans_arg_by_idx(self, idx):
        answers = self.data[idx]['answer']
        res = []
        for answer in answers:
            if type(answer) == str:
                res.append(answer)
            elif type(answer) == dict:
                res.append(answer['answer_argument'])
        return res

    def get_golden_relation_by_idx(self, idx, reverse_prefix=False):
        res = set()
        graph_query = self.get_graph_query(idx)
        for edge in graph_query['edges']:
            e = edge['relation']
            start = edge['start']
            end = edge['end']
            if reverse_prefix and start > end:
                e = 'R ' + e
            res.add(e)
        return list(res)

    def get_len(self):
        return self.len

    def get_s_expression_by_idx(self, idx):
        return get_lisp_from_graph_query(self.get_graph_query(idx))

    def get_golden_entity_by_idx(self, idx, only_id=False, only_label=False):
        res = []
        graph_query = self.get_graph_query(idx)
        for node in graph_query['nodes']:
            if node['node_type'] == 'entity':
                res.append(node)
        if only_id and len(res):
            return [node['id'] for node in res]
        if only_label and len(res):
            return [node['friendly_name'] for node in res]
        return res

    def get_golden_class_by_idx(self, idx, only_label=False):
        res = []
        graph_query = self.get_graph_query(idx)
        for node in graph_query['nodes']:
            if node['node_type'] == 'class':
                res.append(node)
        if only_label and len(res):
            return [node['id'] for node in res]
        return res

    def get_num_entity_by_idx(self, idx):
        return len(self.get_golden_entity_by_idx(idx))

    def get_num_relation_by_idx(self, idx):
        return len(self.get_golden_relation_by_idx(idx))

    def get_num_literal_by_idx(self, idx):
        graph_query = self.get_graph_query(idx)
        res = 0
        for node in graph_query['nodes']:
            if node['node_type'] == 'literal':
                res += 1
        return res

    def get_num_class_by_idx(self, idx):
        return len(self.get_golden_class_by_idx(idx))

    def entity_partial_ratio(self):
        res = total = 0
        s = PartialRatio()
        for idx in range(0, self.len):
            golden_entities = self.get_golden_entity_by_idx(idx, only_label=True)
            question = self.get_question_by_idx(idx).lower()
            for entity in golden_entities:
                res += s.get_sim_score(entity.lower(), question)
                total += 1
        if total != 0:
            return res / total
        return 0

    def class_partial_ratio(self):
        res = total = 0
        s = PartialRatio()
        for idx in range(0, self.len):
            golden_classes = self.get_golden_class_by_idx(idx, only_label=True)
            question = self.get_question_by_idx(idx).lower()
            for c in golden_classes:
                res += s.get_sim_score(c, question)
                total += 1
        if total != 0:
            return res / total
        return 0

    def relation_partial_ratio(self):
        res = total = 0
        s = PartialRatio()
        for idx in range(0, self.len):
            golden_relations = self.get_golden_relation_by_idx(idx)
            question = self.get_question_by_idx(idx).lower()
            for r in golden_relations:
                res += s.get_sim_score(r, question)
                total += 1
        if total != 0:
            return res / total
        return 0


if __name__ == '__main__':
    graphq_train = GraphQuestionsJsonLoader(graphq_train_path)
