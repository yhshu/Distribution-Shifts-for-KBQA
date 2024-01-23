# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy

from utils.domain_dict import fb_domain_dict
from utils.uri_util import remove_ns


def init_graph_query(question_node_class):
    graph_query = {'nodes': [], 'edges': []}
    add_question_node(graph_query, question_node_class)
    return graph_query


def get_num_node(graph_query):
    return len(graph_query['nodes'])


def get_num_edge(graph_query):
    return len(graph_query['edges'])


def get_function(graph_query):
    for node in graph_query['nodes']:
        if node['function'] != 'none':
            return node['function']
    return 'none'


def add_node(graph_query, id, kb_class, friendly_name, node_type='entity', question_node=0, function='none'):
    node = {'nid': get_num_node(graph_query), 'id': id, 'class': kb_class, 'friendly_name': friendly_name,
            'node_type': node_type, 'question_node': question_node, 'function': function}
    graph_query['nodes'].append(node)


def add_question_node(graph_query, id, friendly_name=None, node_type='class', function='none'):
    add_node(graph_query, id, id, friendly_name, node_type=node_type, question_node=1, function=function)


def add_edge(graph_query: dict, start: int, end: int, relation, friendly_name):
    edge = {'start': start, 'end': end, 'relation': relation, 'friendly_name': friendly_name}
    graph_query['edges'].append(edge)


def get_graph_query_function(graph_query: dict):
    for node in graph_query['nodes']:
        if node['function'] != 'none':
            return node['function']
    return 'none'


def get_classes(graph_query: dict):
    res = set()
    for node in graph_query['nodes']:
        if node['node_type'] == 'class':
            res.add(node['class'])
    return res


def get_answer_class(graph_query):
    for node in graph_query['nodes']:
        if node['question_node'] == 1:
            return node['class']
    return None


def get_domain(graph_query):
    answer_class = get_answer_class(graph_query)
    for key in fb_domain_dict:
        if answer_class.startswith(key):
            return key
    return '.'.join(answer_class.split('.')[:2])


def get_entity_id_set(graph_query: dict):
    res = set()
    for node in graph_query['nodes']:
        if node['node_type'] == 'entity':
            if node['id'] is not None and node['id'] != '':
                res.add(node['id'])
    return res


def get_entity_label_list(graph_query: dict):
    res = []
    for node in graph_query['nodes']:
        if node['node_type'] == 'entity':
            if node['friendly_name'] is not None and node['friendly_name'] != '':
                res.append(node['friendly_name'])
    return res


def get_relations(graph_query: dict):
    res = set()
    for edge in graph_query['edges']:
        res.add(edge['relation'])
    return res


def get_terminal_class_nid_list(graph_query: dict):
    res = []
    for node in graph_query['nodes']:
        if node['node_type'] == 'class' and is_terminal_node(graph_query, node['nid']):
            res.append(node['nid'])
    return res


def get_terminal_classes(graph_query: dict):
    res = set()
    for node in graph_query['nodes']:
        if node['node_type'] == 'class' and is_terminal_node(graph_query, node['nid']):
            res.add(node['class'])
    return res


def get_literal_nid_list(graph_query: dict):
    res = []
    for node in graph_query['nodes']:
        if node['node_type'] == 'literal':
            res.append(node['nid'])
    return res


def has_entity_node(graph_query: dict):
    return len(get_entity_id_set(graph_query)) != 0


def has_literal_node(graph_query: dict):
    return len(get_literal_nid_list(graph_query)) != 0


def node_degree(graph_query, nid):
    res = 0
    for edge in graph_query['edges']:
        assert edge['start'] != edge['end']
        if edge['start'] == nid:
            res += 1
        if edge['end'] == nid:
            res += 1
    return res


def replace_var_with_value(graph_query: dict, values: dict, var_label_dict=None):
    for nid in range(len(graph_query['nodes'])):
        node = graph_query['nodes'][nid]
        if node['id'].startswith('?') and node['id'][1:] in values:
            key = node['id'][1:]
            if values[key]['type'] == 'typed-literal':
                graph_query['nodes'][nid]['id'] = values[key]['value'] + '^^' + values[key]['datatype']
            elif values[key]['type'] == 'uri':
                graph_query['nodes'][nid]['id'] = remove_ns(values[key]['value'])

            if var_label_dict is not None and key in var_label_dict:
                graph_query['nodes'][nid]['friendly_name'] = var_label_dict[key]


def is_terminal_node(graph_query, nid):
    return graph_query['nodes'][nid]['question_node'] == 0 and node_degree(graph_query, nid) == 1


def clone_graph_query(graph_query):
    return copy.deepcopy(graph_query)
