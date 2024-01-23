from dataloader.grailqa_json_loader import GrailQAJsonLoader
from utils.config import grailqa_dev_path


def same_schema():
    schema_question_dict = {}
    for idx in range(0, synthetic_data.len):
        golden_class = synthetic_data.get_golden_class_by_idx(idx, True)
        golden_relations = synthetic_data.get_golden_relation_by_idx(idx)
        schema_question_dict[' '.join(golden_class) + ' ' + ' '.join(golden_relations)] = synthetic_data.get_question_by_idx(idx)

    count = 0
    for idx in range(0, grailqa_dev.len):
        golden_class = grailqa_dev.get_golden_class_by_idx(idx, True)
        golden_relations = grailqa_dev.get_golden_relation_by_idx(idx)
        if ' '.join(golden_class) + ' ' + ' '.join(golden_relations) in schema_question_dict:
            count += 1
            print(' '.join(golden_class) + ' ' + ' '.join(golden_relations))
            print(schema_question_dict[' '.join(golden_class) + ' ' + ' '.join(golden_relations)])
            print(grailqa_dev.get_question_by_idx(idx))
            print()
    print(count)


def same_lf():
    lf_question_dict = {}
    for idx in range(0, synthetic_data.len):
        lf_question_dict[synthetic_data.get_s_expression_by_idx(idx)] = synthetic_data.get_question_by_idx(idx)

    count = 0
    for idx in range(0, grailqa_dev.len):
        lf = grailqa_dev.get_s_expression_by_idx(idx)
        if lf in lf_question_dict:
            print(lf)
            print(lf_question_dict[lf])
            print(grailqa_dev.get_question_by_idx(idx))
            count += 1
            print()
    print(count)


if __name__ == '__main__':
    grailqa_dev = GrailQAJsonLoader(grailqa_dev_path)
    synthetic_data = GrailQAJsonLoader('../dataset/GrailQA/question_generation/synthetic_dataset.json')
