import argparse

from tqdm import tqdm

from utils.file_util import read_json_file


def print_synthetic_questions(input_path, output_path):
    with open(output_path, 'w') as f:
        data = read_json_file(input_path)
        for item in tqdm(data):
            print(item['s_expression_with_label'], file=f)
            if len(item['answer']) == 0:
                print('[WARN] no answer')
            print(item['synthetic_question'], file=f)
            print(file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../dataset/question_generation/synthetic_dataset.json')
    parser.add_argument('--output_path', type=str, default='../dataset/question_generation/synthetic_questions.txt')
    args = parser.parse_args()

    print_synthetic_questions(args.input_path, args.output_path)
    print('done')
