# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np


class JsonLoader:
    def get_dataset_split(self):
        if 'train' in self.file_path:
            return 'train'
        elif 'dev' in self.file_path or 'val' in self.file_path or 'vai' in self.file_path:
            return 'dev'
        elif 'test' in self.file_path:
            return 'test'
        elif 'synthetic' in self.file_path:
            return 'synthetic'
        return ''

    def get_golden_relations(self):
        relation_set = set()
        assert self.len is not None and self.len != 0
        for idx in range(0, self.len):
            golden_relation = self.get_golden_relation_by_idx(idx)
            if golden_relation is None or len(golden_relation) == 0:
                continue
            for r in golden_relation:
                relation_set.add(r)
        return relation_set

    def get_golden_relation_frequency(self):
        res = {}
        assert self.len is not None and self.len != 0
        for idx in range(0, self.len):
            golden_relation = self.get_golden_relation_by_idx(idx)
            if golden_relation is None or len(golden_relation) == 0:
                continue
            for r in golden_relation:
                res[r] = res.get(r, 0) + 1
        return res

    def get_num_entity_by_idx(self, idx):
        pass

    def get_num_relation_by_idx(self, idx):
        pass

    def get_num_literal_by_idx(self, idx):
        pass

    def average_question_length(self):
        res = 0
        for idx in range(0, self.len):
            res += len(self.get_question_by_idx(idx))
        if self.len != 0:
            return round(res / self.len, 4)
        return 0

    def std_question_length(self):
        arr = []
        for idx in range(0, self.len):
            arr.append(len(self.get_question_by_idx(idx)))
        if self.len != 0:
            return round(np.std(arr), 4)
        return 0

    def average_num_entity(self):
        res = 0
        for idx in range(0, self.len):
            num = self.get_num_entity_by_idx(idx)
            if num is None:
                return 0
            res += num
        if self.len != 0:
            return round(res / self.len, 4)
        return 0

    def std_num_entity(self):
        arr = []
        for idx in range(0, self.len):
            num = self.get_num_entity_by_idx(idx)
            if num is None:
                return 0
            arr.append(num)
        if self.len != 0:
            return round(np.std(arr), 4)
        return 0

    def average_num_literal(self):
        res = 0
        for idx in range(0, self.len):
            num = self.get_num_literal_by_idx(idx)
            if num is None:
                return 0
            res += num
        if self.len != 0:
            return round(res / self.len, 4)
        return 0

    def std_num_literal(self):
        arr = []
        for idx in range(0, self.len):
            num = self.get_num_literal_by_idx(idx)
            if num is None:
                return 0
            arr.append(num)
        if self.len != 0:
            return round(np.std(arr), 4)
        return 0

    def average_num_relation(self):
        res = 0
        for idx in range(0, self.len):
            num = self.get_num_relation_by_idx(idx)
            if num is None:
                return 0
            res += num
        if self.len != 0:
            return round(res / self.len, 4)
        return 0

    def std_num_relation(self):
        arr = []
        for idx in range(0, self.len):
            num = self.get_num_relation_by_idx(idx)
            if num is None:
                return 0
            arr.append(num)
        if self.len != 0:
            return round(np.std(arr), 4)
        return 0

    def average_num_class(self):
        res = 0
        for idx in range(0, self.len):
            num = self.get_num_class_by_idx(idx)
            if num is None:
                return 0
            res += num
        if self.len != 0:
            return round(res / self.len, 4)
        return 0

    def std_num_class(self):
        arr = []
        for idx in range(0, self.len):
            num = self.get_num_class_by_idx(idx)
            if num is None:
                return 0
            arr.append(num)
        if self.len != 0:
            return round(np.std(arr), 4)
        return 0
