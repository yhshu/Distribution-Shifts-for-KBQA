# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataloader.simple_questions_balance_data_loader import SimpleQuestionsBalanceDataloader
from utils.file_util import write_json_file

if __name__ == '__main__':
    sqb_train = SimpleQuestionsBalanceDataloader('../dataset/SQB/fold-0.train.pickle')
    sqb_dev = SimpleQuestionsBalanceDataloader('../dataset/SQB/fold-0.vaild.pickle')
    sqb_test = SimpleQuestionsBalanceDataloader('../dataset/SQB/fold-0.test.pickle')
    train_relations = sqb_train.get_golden_relations()
    write_json_file('../dataset/SQB/sqb_train.json', sqb_train.to_json(train_relations))
    write_json_file('../dataset/SQB/sqb_dev.json', sqb_dev.to_json(train_relations))
    write_json_file('../dataset/SQB/sqb_test.json', sqb_test.to_json(train_relations))
    print('done')
