# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('rng/framework/')
sys.path.append('GrailQA/')

from dataloader.graphq_json_loader import GraphQuestionsJsonLoader
from dataloader.webqsp_json_loader import WebQSPJsonLoader
from bean.graph_query import get_classes, get_relations
from utils.question_pattern import golden_s_expression_with_golden_entities

import os.path
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, IntervalStrategy, Seq2SeqTrainer

from dataloader.grailqa_json_loader import GrailQAJsonLoader
from retriever.freebase_retriever import FreebaseRetriever
from utils.config import freebase_addr, grailqa_train_path, grailqa_dev_path, grailqa_test_path, freebase_port, graphq_test_path, \
    graphq_pdev_path, graphq_ptrain_path, webqsp_ptrain_path, webqsp_pdev_path, webqsp_test_path
from utils.file_util import pickle_load, pickle_save, write_json_file, read_json_file
from utils.hugging_face_dataset import HFDataset2


class LogicalFormQuestionGeneration():
    def __init__(self, params: dict):
        self.params = params
        self.model_name = params.get('model_name', 't5-base')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        self.max_source_length = 128
        self.max_target_length = 128

        self.num_beams = params.get('num_beams', 10)
        self.train_batch_size = params.get('train_batch_size', 8)
        self.eval_batch_size = params.get('eval_batch_size', 32)

        self.retriever = FreebaseRetriever()

        self.model_eval(params['model_dir'])

    def model_eval(self, model_dir):
        if os.path.isfile(model_dir + '/pytorch_model.bin'):
            self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
            self.model.eval()
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.to(self.device)

    def encode(self, dataloaders: list, output_path, task_prefix='question generation: ', split='train'):
        encodings_path = output_path + '/' + split + '_encodings'
        if output_path is not None and os.path.isfile(encodings_path):
            res = pickle_load(encodings_path)
            return res

        input_sequences = []
        label_sequences = []

        for dataloader in dataloaders:  # for each dataset
            for idx in tqdm(range(0, dataloader.len)):  # for each question
                question = dataloader.get_question_by_idx(idx)
                input_s_expr = golden_s_expression_with_golden_entities(dataloader, idx)
                if input_s_expr is None:
                    continue
                input_sequences.append(input_s_expr)
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

    def train(self, train_dataloaders: list, dev_dataloaders: list, output_dir: str, pretrained_dir=None):
        output_dir = output_dir
        assert output_dir is not None and os.path.isdir(output_dir)
        if os.path.isfile(output_dir + '/pytorch_model.bin'):
            print('Model already exists in {}'.format(output_dir))
            return

        train_encodings = self.encode(train_dataloaders, output_dir.rstrip('/'), split='train')
        dev_encodings = self.encode(dev_dataloaders, output_dir.rstrip('/'), split='dev')

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
                                                 per_device_train_batch_size=self.train_batch_size, per_device_eval_batch_size=self.eval_batch_size, num_train_epochs=10,
                                                 learning_rate=3e-5,
                                                 load_best_model_at_end=False)
        trainer = Seq2SeqTrainer(self.model, args=training_args, train_dataset=train_dataset, eval_dataset=dev_dataset)
        best_run = trainer.train()

        trainer.save_model(output_dir)

    def conditional_generation(self, inputs: list, task_prefix='question generation: '):
        input_ids = self.tokenizer([(task_prefix + x) if x is not None else '' for x in inputs], return_tensors='pt',
                                   max_length=self.max_source_length, truncation=True, padding="max_length").input_ids
        outputs = self.model.generate(input_ids.to(self.device), max_length=self.max_target_length, num_beams=self.num_beams, num_return_sequences=self.num_beams,
                                      output_scores=True, return_dict_in_generate=True)
        outputs_seq = outputs['sequences']
        res = []
        for i in range(len(inputs)):
            res.append(self.tokenizer.decode(outputs_seq[self.num_beams * i], skip_special_tokens=True))
        return res

    def solve(self, dataloaders: list):
        model_inputs = []
        for dataloader in dataloaders:
            for idx in tqdm(range(dataloader.get_len())):  # for each question
                model_inputs.append(golden_s_expression_with_golden_entities(dataloader, idx))

        model_predictions = []
        chunk_size = 100
        all_input_chunks = [model_inputs[i:i + chunk_size] for i in range(0, len(model_inputs), chunk_size)]
        for chunk_idx in tqdm(range(len(all_input_chunks))):
            model_predictions.extend(self.conditional_generation(all_input_chunks[chunk_idx]))

        logs = []
        global_idx = 0
        for dataloader in dataloaders:
            for idx in tqdm(range(dataloader.get_len())):  # for each question
                qid = dataloader.get_question_id_by_idx(idx)
                question = dataloader.get_question_by_idx(idx)
                pred_str = model_predictions[global_idx]
                item = {'qid': qid, 's-expresssion': model_inputs[global_idx], 'question': question, 'synthetic_question': pred_str}

                try:
                    level = dataloader.get_level_by_idx(idx)
                    item['level'] = level
                except Exception as e:
                    pass

                logs.append(item)
                global_idx += 1

        write_json_file('../logs/logical_form_question_generation.json', logs)

    def solve_file(self, file_path: str, comment=None):
        data = read_json_file(file_path)
        model_inputs = [item['s_expression_with_label'] for item in data]
        model_predictions = []

        chunk_size = 100
        all_input_chunks = [model_inputs[i:i + chunk_size] for i in range(0, len(model_inputs), chunk_size)]
        print('#chunk:', len(all_input_chunks), '#sample:', len(model_inputs))
        for chunk_idx in tqdm(range(len(all_input_chunks))):
            model_predictions.extend(self.conditional_generation(all_input_chunks[chunk_idx]))

        for idx in tqdm(range(len(model_inputs))):  # for each question
            pred_str = model_predictions[idx]
            graph_query = data[idx]['graph_query']
            classes = get_classes(graph_query)
            relations = get_relations(graph_query)
            if len(classes.difference(train_golden_classes)) or len(relations.difference(train_golden_relations)):
                data[idx]['level'] = 'zero-shot'
            else:
                data[idx]['level'] = 'seen'
            data[idx]['synthetic_question'] = pred_str

        if comment is None:
            output_path = '../dataset/question_generation/synthetic_dataset_for_lf.json'
        else:
            output_path = '../dataset/question_generation/synthetic_dataset_' + comment + '_for_lf.json'
        write_json_file(output_path, data)
        print('file has been written to ' + output_path)
        # end for each question


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freebase_address', type=str, default=freebase_addr, required=False)
    parser.add_argument('--freebase_port', type=str, default=freebase_port, required=False)
    parser.add_argument('--lf_path', type=str, default='../dataset/question_generation/graph_query_fb.json')
    parser.add_argument('--comment', type=str, default='', required=False)
    parser.add_argument('--model_dir', type=str, default='../model/logical_form_question_generation')
    args = parser.parse_args()

    params = {'freebase_address': args.freebase_address, 'freebase_port': args.freebase_port, 'model_dir': args.model_dir}

    grail_train_data = GrailQAJsonLoader(grailqa_train_path)
    train_golden_classes = grail_train_data.get_golden_classes()
    train_golden_relations = grail_train_data.get_golden_relations()
    grail_dev_data = GrailQAJsonLoader(grailqa_dev_path)
    grail_test_data = GrailQAJsonLoader(grailqa_test_path)

    graphq_ptrain_data = GraphQuestionsJsonLoader(graphq_ptrain_path)
    graphq_pdev_data = GraphQuestionsJsonLoader(graphq_pdev_path)
    graphq_test_data = GraphQuestionsJsonLoader(graphq_test_path)

    webqsp_ptrain_data = WebQSPJsonLoader(webqsp_ptrain_path)
    webqsp_pdev_data = WebQSPJsonLoader(webqsp_pdev_path)
    webqsp_test_data = WebQSPJsonLoader(webqsp_test_path)

    algorithm = LogicalFormQuestionGeneration(params)
    algorithm.train([grail_train_data, graphq_ptrain_data, webqsp_ptrain_data], [grail_dev_data, graphq_pdev_data, webqsp_pdev_data],
                    params['model_dir'])
    if args.lf_path is not None and os.path.isfile(args.lf_path):  # inference for specific file
        algorithm.solve_file(args.lf_path, None if args.comment is None or len(args.comment) == 0 else args.comment)
    else:  # inference for existing KBQA datasets
        algorithm.solve([grail_dev_data, graphq_test_data, webqsp_test_data])
