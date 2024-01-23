# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def is_value_class(c):
    return c == 'type.int' or c == 'type.float' or c == 'type.datetime'


def valid_class(c):
    return c not in {'type.boolean', 'type.enumeration', 'type.uri', 'type.text', 'type.rawstring'}
