# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys

sys.path.append('rng/framework/')

from rng.WebQSP.enumerate_candidates import get_approx_s_expr
from rng.WebQSP.eval_topk_prediction import get_time_macro_clause
from rng.framework.executor.sparql_executor import execute_query
from GrailQA.utils.logic_form_util import lisp_to_sparql


def execute_s_expr(expr):
    if 'time_macro' in expr:
        try:
            approx_expr = get_approx_s_expr(expr)
        except:
            return 'null', []
        try:
            additional_clause = get_time_macro_clause(expr)
            approx_sparql = lisp_to_sparql(approx_expr)
            approx_sparql_end = approx_sparql.rfind('}')
            cat_sqarql = approx_sparql[:approx_sparql_end] + additional_clause + approx_sparql[approx_sparql_end:]

            cat_result = execute_query(cat_sqarql)

            return expr, cat_result
        except Exception as e:
            print('execute_s_expr exception', e)
            return 'null', []
    else:
        # query_expr = expr.replace('( ', '(').replace(' )', ')')
        # return query_expr, []
        try:
            # print('parse', query_expr)
            sparql_query = lisp_to_sparql(expr)
            # print('sparql', sparql_query)
            denotation = execute_query(sparql_query)
        except Exception as e:
            print('execute_s_expr exception', e)
            return 'null', []
        return expr, denotation


def get_function_in_lisp(lisp: str):
    if 'COUNT' in lisp:
        return 'count'
    elif 'ARGMIN' in lisp:
        return 'argmin'
    elif 'ARGMAX' in lisp:
        return 'argmax'
    elif 'TC' in lisp:
        return 'tc'
    elif 'le' in lisp:
        return 'le'
    elif 'ge' in lisp:
        return 'ge'
    elif 'lt' in lisp:
        return 'lt'
    elif 'gt' in lisp:
        return 'gt'
    return 'none'
