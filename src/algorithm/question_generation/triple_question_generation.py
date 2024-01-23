# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('dataloader')

from dataloader.simple_questions_balance_data_loader import SimpleQuestionsBalanceDataloader
import argparse
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, IntervalStrategy
from retriever.freebase_retriever import FreebaseRetriever
from utils.config import freebase_port, freebase_addr, sqb_train_path, sqb_dev_path, sqb_test_path
from utils.file_util import write_json_file, pickle_load, pickle_save, read_json_file
from utils.hugging_face_dataset import HFDataset2


class SimpleQuestionsBalanceQuestionGeneration():

    def __init__(self, params: dict):
        self.params = params
        self.model_name = params.get('model_name', 't5-base')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        self.max_source_length = 32
        self.max_target_length = 32

        self.num_beams = params.get('num_beams', 10)
        self.train_batch_size = params.get('train_batch_size', 8)
        self.eval_batch_size = params.get('eval_batch_size', 32)

        self.retriever = FreebaseRetriever()

        self.model_eval(params.get('model_dir', self.model_name))

    def model_eval(self, model_dir):
        if os.path.isfile(model_dir + '/pytorch_model.bin'):
            self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
            self.model.eval()
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.to(self.device)

    def conditional_generation(self, inputs: list, task_prefix='question generation: '):
        input_ids = self.tokenizer([task_prefix + x for x in inputs], return_tensors='pt', max_length=self.max_source_length, truncation=True, padding="max_length").input_ids
        outputs = self.model.generate(input_ids.to(self.device), max_length=self.max_target_length, num_beams=self.num_beams, num_return_sequences=self.num_beams,
                                      output_scores=True, return_dict_in_generate=True)
        outputs_seq = outputs['sequences']
        res = []
        for i in range(len(inputs)):
            res.append(self.tokenizer.decode(outputs_seq[self.num_beams * i], skip_special_tokens=True))
        return res

    def encode(self, dataloader: SimpleQuestionsBalanceDataloader, output_path, task_prefix='question generation: '):
        encodings_path = output_path + '/' + dataloader.get_dataset_split() + '_encodings'
        if output_path is not None and os.path.isfile(encodings_path):
            res = pickle_load(encodings_path)
            return res

        input_sequences = []
        label_sequences = []

        for idx in tqdm(range(0, dataloader.len)):  # for each question
            question = dataloader.get_question_by_idx(idx)
            triple_text = dataloader.get_subject_predicate_text_by_idx(idx)
            input_sequences.append(triple_text)
            label_sequences.append(question)
        # end for each question

        print('encoding ', len(input_sequences))
        # encode the inputs
        encodings = self.tokenizer([task_prefix + sequence for sequence in input_sequences], padding='max_length', max_length=self.max_source_length, truncation=True)
        input_ids, attention_mask = encodings.input_ids, encodings.attention_mask
        # print example
        print('[Example]')
        for i in range(0, 5):
            print('[Input]' + input_sequences[i])
            print('[Output]' + label_sequences[i])
            print('[Input token len]' + str(len(self.tokenizer.tokenize(input_sequences[i]))))
            print('\n')

        res = {'input_ids': input_ids, 'attention_mask': attention_mask}

        # encode the targets
        if len(label_sequences):
            target_encoding = self.tokenizer(label_sequences, padding='max_length', max_length=self.max_target_length, truncation=True)
            labels = target_encoding.input_ids

            # replace padding token id's of the labels by -100
            labels = torch.tensor(labels)
            labels[labels == self.tokenizer.pad_token_id] = -100
            res['labels'] = labels

        pickle_save(res, encodings_path)
        return res

    def train(self, train_dataloader, dev_dataloader, output_dir: str, pretrained_dir=None):
        assert output_dir is not None and os.path.isdir(output_dir)
        if os.path.isfile(output_dir + '/pytorch_model.bin'):
            print('Model already exists in {}'.format(output_dir))
            return

        train_encodings = self.encode(train_dataloader, output_dir.rstrip('/'))
        dev_encodings = self.encode(dev_dataloader, output_dir.rstrip('/'))

        train_dataset = HFDataset2(train_encodings)
        dev_dataset = HFDataset2(dev_encodings)

        # training settings
        if pretrained_dir is None:
            pretrained_dir = self.model_name
        else:
            assert os.path.isdir(pretrained_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_dir)
        self.model.train()

        training_args = Seq2SeqTrainingArguments(output_dir=output_dir, do_train=True, do_eval=True, do_predict=True,
                                                 # evaluation_strategy=IntervalStrategy.STEPS, save_strategy=IntervalStrategy.STEPS, eval_steps=500,
                                                 # eval_accumulation_steps=200,
                                                 evaluation_strategy=IntervalStrategy.EPOCH, save_strategy=IntervalStrategy.EPOCH,
                                                 per_device_train_batch_size=self.train_batch_size, per_device_eval_batch_size=self.eval_batch_size, num_train_epochs=8,
                                                 learning_rate=3e-5,
                                                 load_best_model_at_end=False)
        trainer = Seq2SeqTrainer(self.model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset)
        best_run = trainer.train()

        trainer.save_model(output_dir)

    def solve_file(self, file_path: str, comment=None):
        data = read_json_file(file_path)
        model_inputs = [item[3] + '|' + item[1] for item in data]
        print('#relation', len(set(item[1] for item in data)))
        print('#sample', len(data))
        model_predictions = []

        chunk_size = 250
        all_input_chunks = [model_inputs[i:i + chunk_size] for i in range(0, len(model_inputs), chunk_size)]
        print('#chunk:', len(all_input_chunks), 'chunk_size:', chunk_size, '#sample:', len(model_inputs))
        for chunk_idx in tqdm(range(len(all_input_chunks))):
            model_predictions.extend(self.conditional_generation(all_input_chunks[chunk_idx]))

        res = []
        for idx in tqdm(range(len(model_inputs))):  # for each question
            res.append({'subject': data[idx][0], 'relation': data[idx][1], 'object': data[idx][2],
                        'subject_text': data[idx][3], 'object_text': data[idx][4],
                        'synthetic_question': model_predictions[idx]})

        if comment is None:
            output_path = '../dataset/question_generation/synthetic_dataset_for_triple.json'
        else:
            output_path = '../dataset/SQB/question_generation/synthetic_dataset_' + comment + '_for_triple.json'
        write_json_file(output_path, res)
        print('file has been written to ' + output_path)
        # end for each question

    def solve(self, dataloader: SimpleQuestionsBalanceDataloader):
        model_inputs = []
        for idx in tqdm(range(dataloader.get_len())):  # for each question
            triple_text = dataloader.get_subject_predicate_text_by_idx(idx)
            model_inputs.append(triple_text)

        model_predictions = []
        chunk_size = 100
        all_input_chunks = [model_inputs[i:i + chunk_size] for i in range(0, len(model_inputs), chunk_size)]
        for chunk_idx in tqdm(range(len(all_input_chunks))):
            model_predictions.extend(self.conditional_generation(all_input_chunks[chunk_idx]))

        logs = []
        for idx in tqdm(range(dataloader.get_len())):  # for each question
            question = dataloader.get_question_by_idx(idx)
            pred_str = model_predictions[idx]
            logs.append({'triple_text': model_inputs[idx], 'question': question, 'synthetic_question': pred_str})

        write_json_file('../logs/triple_question_generation.json', logs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freebase_address', type=str, default=freebase_addr, required=False)
    parser.add_argument('--freebase_port', type=str, default=freebase_port, required=False)
    parser.add_argument('--triple_path', type=str, default='../dataset/question_generation/triple_fb.json')
    parser.add_argument('--comment', type=str, default='', required=False)
    parser.add_argument('--model_dir', type=str, default='../model/triple_question_generation')

    args = parser.parse_args()

    params = {'freebase_address': args.freebase_address, 'freebase_port': args.freebase_port,
              'golden_entity': 'false', 'golden_schema': 'false', 'model_dir': args.model_dir}

    sqb_train = SimpleQuestionsBalanceDataloader(sqb_train_path)
    sqb_dev = SimpleQuestionsBalanceDataloader(sqb_dev_path)
    sqb_test = SimpleQuestionsBalanceDataloader(sqb_test_path)

    algorithm = SimpleQuestionsBalanceQuestionGeneration(params)
    algorithm.train(sqb_train, sqb_dev, args.model_dir)
    if args.triple_path is not None and os.path.isfile(args.triple_path):
        print('solving file:', args.triple_path)
        algorithm.solve_file(args.triple_path, None if args.comment is None or len(args.comment) == 0 else args.comment)
    else:
        algorithm.solve(sqb_dev)
