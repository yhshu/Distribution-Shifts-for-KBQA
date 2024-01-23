# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('GrailQA')

from GrailQA.utils.logic_form_util import lisp_to_sparql, get_lisp_from_graph_query, reverse_properties
import argparse
from algorithm.graph_query.enumeration_utils import valid_class, is_value_class
import uuid
import random
from tqdm import tqdm
from bean.graph_query import init_graph_query, clone_graph_query, add_node, add_edge, get_num_node, get_num_edge, get_terminal_class_nid_list, get_classes, get_literal_nid_list, \
    replace_var_with_value, get_relations, get_graph_query_function
from retriever.freebase_retriever import FreebaseRetriever
from utils.config import grailqa_classes_path, grailqa_relations_path, grailqa_entity_classes_path, fb_entity_classes_path
from utils.file_util import read_set_file, write_json_file
from utils.s_expr_util import execute_s_expr
from utils.uri_util import is_mid_gid, schema_domain


def expand_graph_query(graph_query, start, cvt=True, class_node=True, literal_node=True, domains=None):
    """
    Expand a graph query from the start node to the end node
    :param cvt:
    :param graph_query: a graph query
    :param start: start node id
    :return: expanded graph query
    """
    res = []
    if start >= len(graph_query['nodes']):
        return res
    c = graph_query['nodes'][start]['class']

    if c in domain_to_relations:  # relation from c to another entity / literal / class
        for relation, relation_range in domain_to_relations[c]:
            if cvt is False and relation_range not in candidate_entity_classes and not is_value_class(relation_range):
                continue
            if domains is not None and schema_domain(relation) not in domains:
                continue

            node_type = 'literal' if is_value_class(relation_range) else 'class'
            if (node_type == 'literal' and literal_node is False) or (node_type == 'class' and class_node is False):
                continue
            if node_type == 'literal' and relation in get_relations(graph_query):  # the literal relation only appears once
                continue
            new_graph_query = clone_graph_query(graph_query)
            add_node(new_graph_query, id=relation_range, kb_class=relation_range, friendly_name=None, node_type=node_type, question_node=0, function='none')
            add_edge(new_graph_query, start, get_num_node(new_graph_query) - 1, relation, friendly_name=None)

            if cvt is True and relation_range not in candidate_entity_classes and node_type == 'class' and relation_range in domain_to_relations:  # add a node for CVT
                for cvt_relation, cvt_relation_range in domain_to_relations[relation_range]:
                    if cvt_relation_range not in candidate_entity_classes and not is_value_class(cvt_relation_range):  # only one CVT
                        continue
                    cvt_node_type = 'literal' if is_value_class(cvt_relation_range) else 'class'
                    if (cvt_node_type == 'literal' and literal_node is False) or (cvt_node_type == 'class' and class_node is False):
                        continue
                    if cvt_node_type == 'literal' and cvt_relation_range in get_classes(new_graph_query):  # unique literal class
                        continue
                    cvt_graph_query = clone_graph_query(new_graph_query)
                    add_node(cvt_graph_query, id=cvt_relation_range, kb_class=cvt_relation_range, friendly_name=None, node_type=cvt_node_type, question_node=0, function='none')
                    num_node = get_num_node(cvt_graph_query)
                    add_edge(cvt_graph_query, num_node - 2, num_node - 1, cvt_relation, friendly_name=None)
                    res.append(cvt_graph_query)
            else:  # no CVT
                res.append(new_graph_query)
    return res


def expand_graph_query_list(graph_query_list, start, cvt, edge_upper_bound, class_node=True, literal_node=True, domains=None, sample=True):
    res = []
    for gq in graph_query_list:
        if sample is True:
            prob = 1.0
            if edge_upper_bound == 3:
                prob = 0.07
            if prob < 1.0:
                skip = random.choices(['keep', 'skip'], [prob, 1.0 - prob])[0]
                if skip == 'skip':
                    continue
        res.extend(expand_graph_query(gq, start, cvt, class_node, literal_node, domains))
    return res


def english_filters(sparql: str, vars: list):
    filters = []
    for var in vars:
        filters += 'FILTER (!isLiteral(' + var + ') OR lang(' + var + ') = \'\' OR langMatches(lang(' + var + '), \'en\'))\n'
    assert 'WHERE {\n' in sparql
    return sparql.replace('WHERE {\n', 'WHERE {\n' + ''.join(filters) + ' ')


def graph_query_for_class_and_edges(c: str, edge_upper_bound, level='zero-shot', verbose=True):
    """
    Generate graph queries for a KB class
    :param c: a KB class
    :param num_instance: number of entity instances to be generated
    :param level:  generalization level
    :return: graph queries
    """

    # generate query template
    assert 1 <= edge_upper_bound <= 3
    empty_graph_query = init_graph_query(c)
    ungrounded_graph_queries = []
    if edge_upper_bound == 1:
        num_instance = 10
        ungrounded_graph_queries = expand_graph_query(empty_graph_query, 0, cvt=False)
    elif edge_upper_bound == 2:  # 0 -> 1, 1 -> 2; 0 -> 1, 0 -> 2
        num_instance = 1
        ungrounded_graph_queries = expand_graph_query(empty_graph_query, 0, cvt=True)
        t = expand_graph_query(empty_graph_query, 0, cvt=False)
        ungrounded_graph_queries.extend(expand_graph_query_list(t, 0, cvt=False, edge_upper_bound=2))
    elif edge_upper_bound == 3:
        num_instance = 1
        gq02 = expand_graph_query(empty_graph_query, 0, cvt=True)
        ungrounded_graph_queries.extend(expand_graph_query_list(gq02, 2, cvt=False, edge_upper_bound=3))
        ungrounded_graph_queries.extend(expand_graph_query_list(gq02, 1, cvt=False, edge_upper_bound=3))
        ungrounded_graph_queries.extend(expand_graph_query_list(gq02, 0, cvt=False, edge_upper_bound=3))
        gq01 = expand_graph_query(empty_graph_query, 0, cvt=False)
        ungrounded_graph_queries.extend(expand_graph_query_list(gq01, 1, cvt=True, edge_upper_bound=3))
        gq0102 = expand_graph_query_list(gq01, 0, cvt=False, edge_upper_bound=3)
        ungrounded_graph_queries.extend(expand_graph_query_list(gq0102, 0, cvt=False, edge_upper_bound=3))

    # determine template node
    grounded_graph_queries = []
    for graph_query in ungrounded_graph_queries:
        if c in ['measurement_unit.measurement_system']:  # down sampling for classes with large number of graph queries
            skip = random.choices(['skip', 'keep'], [0.85, 0.15])[0]
            if skip == 'skip':
                continue

        num_node = get_num_node(graph_query)
        num_edge = get_num_edge(graph_query)
        assert 2 <= num_node <= 4 and num_node == num_edge + 1

        if num_edge == 3 and len(get_relations(graph_query)) < 3:
            continue

        terminal_class_nid_list = get_terminal_class_nid_list(graph_query)
        if len(terminal_class_nid_list) > 2:
            continue
        entity_var_list = []
        invalid = False
        for nid in terminal_class_nid_list:
            if graph_query['nodes'][nid]['class'] not in candidate_entity_classes:
                invalid = True
                if verbose:
                    print('invalid terminal class:', graph_query['nodes'][nid]['class'])
                break
            graph_query['nodes'][nid]['node_type'] = 'entity'
            graph_query['nodes'][nid]['id'] = '?e' + str(nid)
            entity_var_list.append('?e' + str(nid))

        if invalid:
            continue

        literal_nid_list = get_literal_nid_list(graph_query)
        if len(literal_nid_list) > 2:
            continue
        literal_var_list = []
        for nid in literal_nid_list:
            if graph_query['nodes'][nid]['node_type'] == 'literal':
                graph_query['nodes'][nid]['id'] = '?l' + str(nid)
                literal_var_list.append('?l' + str(nid))

        temp_grounded_graph_queries = []
        if len(entity_var_list):  # grond entity nodes
            s_expr = get_lisp_from_graph_query(graph_query)
            sparql = lisp_to_sparql(s_expr)

            entity_sparql = sparql.replace('FILTER (!isLiteral(?x) OR lang(?x) = \'\' OR langMatches(lang(?x), \'en\'))', '') \
                .replace('SELECT DISTINCT ?x', 'SELECT DISTINCT ' + ' '.join(entity_var_list))
            entity_sparql = english_filters(entity_sparql, entity_var_list) + ' LIMIT 10'
            ret = retriever.query(entity_sparql)  # (s_expression, result)
            random.shuffle(ret)
            count_instance = 0
            for r in ret:  # for each execution result
                var_label_dict = {}
                for var in r:  # for each variable
                    label = retriever.rdf_label_by_mid(r[var]['value'], True)
                    if label is None or len(label) == 0:
                        break
                    var_label_dict[var] = label
                if len(var_label_dict) < len(r):  # there's entity without label, this r is not abandoned
                    continue
                entity_id_set = set([r[key]['value'] for key in r])
                if len(entity_id_set) < len(r):  # there's duplicate entities
                    continue
                new_graph_query = clone_graph_query(graph_query)
                replace_var_with_value(new_graph_query, r, var_label_dict)
                temp_grounded_graph_queries.append(new_graph_query)
                count_instance += 1
                if count_instance >= num_instance:
                    break
        else:
            temp_grounded_graph_queries = [graph_query]

        if len(literal_var_list):  # ground literal nodes
            for t_graph_query in temp_grounded_graph_queries:
                s_expr = get_lisp_from_graph_query(t_graph_query)
                sparql = lisp_to_sparql(s_expr)

                literal_sparql = sparql.replace('FILTER (!isLiteral(?x) OR lang(?x) = \'\' OR langMatches(lang(?x), \'en\'))', '') \
                                     .replace('SELECT DISTINCT ?x', 'SELECT DISTINCT ' + ' '.join(literal_var_list)) + ' LIMIT 10'
                ret = retriever.query(literal_sparql)
                random.shuffle(ret)
                count_instance = 0
                for r in ret:
                    literal_value_set = set([r[key]['value'] for key in r])
                    if len(literal_value_set) < len(r):  # there's duplicate literals
                        continue
                    new_graph_query = clone_graph_query(t_graph_query)
                    replace_var_with_value(new_graph_query, r)
                    grounded_graph_queries.append(new_graph_query)
                    count_instance += 1
                    if count_instance >= num_instance:
                        break
        else:
            grounded_graph_queries.extend(temp_grounded_graph_queries)

    grounded_with_function = []  # ungrounded graph queries with function
    for graph_query in grounded_graph_queries:  # graph query with function
        add_count = random.choices(['none', 'count'], [0.94, 0.06])[0]
        if add_count == 'count':
            new_graph_query = clone_graph_query(graph_query)
            new_graph_query['nodes'][0]['function'] = add_count
            grounded_with_function.append(new_graph_query)

        literal_nid_list = get_literal_nid_list(graph_query)
        if len(literal_nid_list):
            add_superlative = random.choices(['none', 'argmin', 'argmax'], [0.88, 0.06, 0.06])[0]
            if add_superlative != 'none':
                new_graph_query = clone_graph_query(graph_query)
                selected_nid = random.sample(literal_nid_list, 1)[0]
                new_graph_query['nodes'][selected_nid]['function'] = add_superlative
                id = new_graph_query['nodes'][selected_nid]['id']
                new_graph_query['nodes'][selected_nid]['id'] = '0' + id[id.index('^^'):]
                grounded_with_function.append(new_graph_query)

            add_comparative = random.choices(['none', '<=', '<', '>=', '>'], [0.86, 0.03, 0.04, 0.03, 0.04])[0]
            if add_comparative != 'none':
                new_graph_query = clone_graph_query(graph_query)
                selected_nid = random.sample(literal_nid_list, 1)[0]
                new_graph_query['nodes'][selected_nid]['function'] = add_comparative
                grounded_with_function.append(new_graph_query)

    grounded_graph_queries.extend(grounded_with_function)

    # reverse property
    grounded_reversed = []
    for graph_query in grounded_graph_queries:
        new_graph_query = clone_graph_query(graph_query)
        changed = False
        for eid in range(get_num_edge(graph_query)):
            r = graph_query['edges'][eid]['relation']
            if r not in reverse_properties or random.choices(['true', 'false'], [0.2, 0.8])[0] == 'false':
                continue
            changed = True
            new_graph_query['edges'][eid]['start'] = graph_query['edges'][eid]['end']
            new_graph_query['edges'][eid]['end'] = graph_query['edges'][eid]['start']
            new_graph_query['edges'][eid]['relation'] = reverse_properties[r]
        if changed:
            grounded_reversed.append(new_graph_query)
    grounded_graph_queries.extend(grounded_reversed)

    # validate generated graph queries
    res = []
    for graph_query in grounded_graph_queries:
        s_expr = get_lisp_from_graph_query(graph_query)
        sparql = lisp_to_sparql(s_expr)
        function = get_graph_query_function(graph_query)
        t = execute_s_expr(s_expr)  # (s_expression, result)
        if t[0] == 'null':
            print('Converting S-expression to SPARQL, but null:', s_expr)
        ans = t[1]
        if len(ans):  # there's result for this query
            has_ans = True
            if function != 'count':
                for a in ans[:1]:  # assume an answer must be an entity
                    label = retriever.rdf_label_by_mid(a, only_one=True)
                    if len(label) == 0 or is_mid_gid(label):  # no entity label
                        has_ans = False
                        break
                if not has_ans:  # no answer for this query
                    if verbose:
                        print('No answer for this query:', s_expr)
                    continue
            else:  # count
                assert len(ans) == 1
                if ans[0] == '0':
                    if verbose:
                        print('No answer for this query:', s_expr)
                    continue

            s_expr_with_label = s_expr
            mid_labels = []
            for node in graph_query['nodes']:
                if node['node_type'] == 'entity':
                    friendly_name = retriever.rdf_label_by_mid(node['id'], only_one=True) if node['friendly_name'] is None else node['friendly_name']
                    mid_labels.append((node['id'], friendly_name))
            if len(mid_labels):
                s_expr_with_label += '|entity'
                for mid, label in mid_labels:
                    s_expr_with_label += '|' + mid + ' ' + label.lower()
            res.append({'qid': str(uuid.uuid4()), 'level': level, 'graph_query': graph_query, 's_expression': s_expr,
                        's_expression_with_label': s_expr_with_label, 'sparql': sparql, 'answer': ans})
    if verbose:
        print(c, 'edge:', edge_upper_bound, '#grounded:', len(grounded_graph_queries), '#valid:', len(res))
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=429)
    parser.add_argument('--domain', type=str, default='grailqa')
    parser.add_argument('--output_dir', type=str, default='../dataset/question_generation')
    args = parser.parse_args()

    # preparation
    sample_domain = args.domain
    random.seed(args.seed)
    retriever = FreebaseRetriever()

    grailqa_classes = read_set_file(grailqa_classes_path)
    grailqa_entity_classes = read_set_file(grailqa_entity_classes_path)
    fb_entity_classes = read_set_file(fb_entity_classes_path)
    grailqa_relations = read_set_file(grailqa_relations_path)
    grailqa_train_domains = read_set_file('../dataset/GrailQA/grailqa_train_domain.txt')
    candidate_entity_classes = grailqa_entity_classes if sample_domain == 'grailqa' else fb_entity_classes
    with open('../dataset/GrailQA/ontology/fb_roles', 'r') as f:
        fb_roles = f.readlines()
    domain_to_relations = {}
    range_to_relations = {}
    for line in fb_roles:
        line_split = line.strip('\n').split(' ')
        relation_domain = line_split[0]
        relation = line_split[1]
        relation_range = line_split[2]
        if sample_domain == 'grailqa' and (relation_domain not in grailqa_classes or relation_range not in grailqa_classes or relation not in grailqa_relations):
            continue
        if not valid_class(relation_domain) or not valid_class(relation_range):
            continue

        if relation_domain not in domain_to_relations:
            domain_to_relations[relation_domain] = set()
        domain_to_relations[relation_domain].add((relation, relation_range))
        if relation_range not in range_to_relations:
            range_to_relations[relation_range] = set()
        range_to_relations[relation_range].add((relation, relation_domain))

    # output path
    s_expression_with_label_file_path = args.output_dir + '/s_expression.txt'
    s_expression_with_label_file = open(s_expression_with_label_file_path, 'w')
    graph_query_file_path = args.output_dir + '/graph_query_fb.json'
    res = []

    print('sampling domain', sample_domain)
    print('#class', len(candidate_entity_classes))
    for schema in tqdm(candidate_entity_classes):
        if is_value_class(schema):
            continue
        level = 'seen_domain' if schema_domain(schema) in grailqa_train_domains else 'zero-shot'
        for num_edge in [1, 2]:
            graph_queries = graph_query_for_class_and_edges(schema, num_edge, level=level)
            res.extend(graph_queries)
            for g in graph_queries:
                print(g['s_expression_with_label'], file=s_expression_with_label_file)

    write_json_file(graph_query_file_path, res)
    s_expression_with_label_file.close()
