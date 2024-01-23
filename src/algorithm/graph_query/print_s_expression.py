# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')

import argparse

from utils.file_util import read_json_file


def output_s_expr(graph_query_path: str, output_path: str):
    graph_queries = read_json_file(graph_query_path)
    with open(output_path, 'w') as f:
        for graph_query in graph_queries:
            print(graph_query['s_expression_with_label'], file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_query_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    output_s_expr(args.graph_query_path, args.output_path)
    print('done')
