# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from retriever.freebase_retriever import FreebaseRetriever
from utils.uri_util import is_mid_gid


def freebase_entity_classes():
    with open('../dataset/GrailQA/ontology/fb_roles', 'r') as f:
        fb_roles = f.readlines()

    classes = set()
    for line in fb_roles:
        line_split = line.strip('\n').split(' ')
        relation_domain = line_split[0]
        relation = line_split[1]
        relation_range = line_split[2]

        classes.add(relation_domain)
        classes.add(relation_range)

    for c in classes:
        instances = retriever.instance_by_uri(c, sample=False)
        for entity in instances:
            labels = retriever.rdf_label_by_mid(entity, True)
            if len(labels) != 0 and not is_mid_gid(labels):
                print(c)
                break


if __name__ == '__main__':
    retriever = FreebaseRetriever()
    freebase_entity_classes()
