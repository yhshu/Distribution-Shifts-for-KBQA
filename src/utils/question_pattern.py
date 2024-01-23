# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataloader.webqsp_json_loader import WebQSPJsonLoader


def s_expression_with_labels(s_expr, entities, lowercase=True, simplify_literal_prefix=False):
    res = s_expr
    if s_expr is None:
        return res
    if entities is not None and len(entities):
        mid_list = [entity['id'] for entity in entities if entity['id'] in s_expr]
        if len(mid_list):
            res += '|entity'
        for entity in entities:
            if entity['id'] not in s_expr:
                continue
            if 'friendly_name' in entity:
                name = entity['friendly_name']
            else:
                name = entity['label']
            if lowercase:
                name = name.lower()
            res += '|' + entity['id'] + ' ' + name
    if simplify_literal_prefix:
        res = res.replace('^^http://www.w3.org/2001/XMLSchema#', '^^')
    return res


def golden_s_expression_with_golden_entities(dataloader, idx, lowercase=True):
    golden_s_expression = dataloader.get_s_expression_by_idx(idx)
    if golden_s_expression is None:
        print(type(dataloader), idx, 'golden_s_expression is None')
    if isinstance(dataloader, WebQSPJsonLoader):
        golden_entities = dataloader.get_golden_entity_by_idx(idx, constraint=True, return_dict_list=True)
    else:
        golden_entities = dataloader.get_golden_entity_by_idx(idx)
    return s_expression_with_labels(golden_s_expression, golden_entities, lowercase)
