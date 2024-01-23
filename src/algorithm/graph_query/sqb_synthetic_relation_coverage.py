# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

from utils.config import sqb_test_path, sqb_dev_path

sys.path.append('dataloader')

from dataloader.simple_questions_balance_data_loader import SimpleQuestionsBalanceDataloader
from utils.file_util import read_set_file

if __name__ == '__main__':
    sqb_dev = SimpleQuestionsBalanceDataloader(sqb_dev_path)
    sqb_test = SimpleQuestionsBalanceDataloader(sqb_test_path)
    fb_roles_relations = read_set_file('../dataset/fb_roles_relations.txt')
    dev_relations = sqb_dev.get_golden_relations()
    test_relations = sqb_test.get_golden_relations()
    print(len(dev_relations.difference(fb_roles_relations)) / len(dev_relations))
    print(len(test_relations.difference(fb_roles_relations)) / len(test_relations))

    dev_miss = test_miss = 0
    for idx in range(0, sqb_dev.len):
        golden_relation = sqb_dev.get_golden_relation_by_idx(idx)
        if golden_relation not in fb_roles_relations:
            dev_miss += 1
    for idx in range(0, sqb_test.len):
        golden_relation = sqb_test.get_golden_relation_by_idx(idx)
        if golden_relation not in fb_roles_relations:
            test_miss += 1
    print(dev_miss / sqb_dev.len, sqb_dev.len, test_miss / sqb_test.len, sqb_test.len)
