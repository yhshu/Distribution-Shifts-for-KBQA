# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')
from py_stringmatching import PartialRatio
from utils.config import sqb_test_path, sqb_dev_path, sqb_train_path
from utils.file_util import read_json_file
from dataloader.json_loader import JsonLoader
import pickle as pkl


class SimpleQuestionsBalanceDataloader(JsonLoader):
    def __init__(self, file_path: str, synthetic=False):
        self.file_path = file_path
        self.synthetic = synthetic
        if not synthetic and not file_path.endswith('.json'):
            self.data = pkl.load(open(file_path, 'rb'))
        else:
            self.data = read_json_file(file_path)
        self.relation_vocab = pkl.load(open('../dataset/SQB/Embedding/rel.voc.pickle', 'rb'))
        self.len = len(self.data)

    def get_len(self):
        return self.len

    def get_question_by_idx(self, idx):
        if self.synthetic:
            return self.data[idx]['synthetic_question']
        if isinstance(self.data[idx], dict):
            return self.data[idx]['question']
        return self.data[idx].question

    def get_question_id_by_idx(self, idx):
        return self.data[idx].qid

    def get_sparql_by_idx(self, idx):
        return None

    def get_anonymous_question_by_idx(self, idx):
        assert self.synthetic is False
        return self.data[idx].anonymous_question

    def get_golden_relation_by_idx(self, idx):
        if isinstance(self.data[idx], dict):
            return self.data[idx]['relation']
        return self.relation_vocab[self.data[idx].relation]

    def get_golden_relations(self):
        res = set()
        for idx in range(0, self.len):
            res.add(self.get_golden_relation_by_idx(idx))
        return res

    def get_golden_entity_by_idx(self, idx):
        if isinstance(self.data[idx], dict):
            return self.data[idx]['subject']
        return self.data[idx].subject

    def get_ans_id_by_idx(self, idx):
        if isinstance(self.data[idx], dict):
            return self.data[idx]['object']
        return self.data[idx].object

    def get_golden_entity_text_by_idx(self, idx):
        if isinstance(self.data[idx], dict):
            return self.data[idx]['subject_text']
        return self.data[idx].subject_text

    def get_subject_predicate_text_by_idx(self, idx):
        if self.synthetic:
            return self.data[idx]['subject_text'] + '|' + self.data[idx]['relation']
        return self.data[idx].subject_text + '|' + self.get_golden_relation_by_idx(idx)

    def get_subject_id_and_predicate_by_idx(self, idx):
        if self.synthetic:
            return self.data[idx]['subject'] + '|' + self.data[idx]['relation']
        return self.data[idx].subject + '|' + self.get_golden_relation_by_idx(idx)

    def get_diverse_score_by_idx(self, idx):
        if 'diverse_score' in self.data[idx]:
            return float(self.data[idx]['diverse_score'])
        return None

    def to_json(self, train_relations):
        res = []
        for idx in range(0, self.len):
            qid = self.get_question_id_by_idx(idx)
            relation = self.get_golden_relation_by_idx(idx)
            res.append({'qid': qid, 'question': self.get_question_by_idx(idx),
                        'anonymous_question': self.get_anonymous_question_by_idx(idx),
                        'subject': self.get_golden_entity_by_idx(idx), 'subject_text': self.get_golden_entity_text_by_idx(idx),
                        'relation': relation, 'object': self.get_ans_id_by_idx(idx),
                        'level': 'seen' if relation in train_relations else 'unseen'})
        print('len:', len(res))
        return res

    def entity_partial_ratio(self):
        res = total = 0
        s = PartialRatio()
        for idx in range(0, self.len):
            golden_entities = self.get_golden_entity_text_by_idx(idx)
            question = self.get_question_by_idx(idx).lower()
            for entity in golden_entities:
                res += s.get_sim_score(entity.lower(), question)
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
    sqb_train = SimpleQuestionsBalanceDataloader(sqb_train_path)
    print(sqb_train.len)

    sqb_dev = SimpleQuestionsBalanceDataloader(sqb_dev_path)
    print(sqb_dev.len)

    sqb_test = SimpleQuestionsBalanceDataloader(sqb_test_path)
    print(sqb_test.len)
