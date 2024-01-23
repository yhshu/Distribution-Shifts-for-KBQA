# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from retriever.freebase_retriever import FreebaseRetriever
from utils.config import grailqa_classes_path
from utils.file_util import read_set_file


def grailqa_entity_classes():
    grailqa_classes = read_set_file(grailqa_classes_path)

    for c in grailqa_classes:
        instances = retriever.instance_by_uri(c, sample=False)
        for entity in instances:
            labels = retriever.rdf_label_by_mid(entity)
            if len(labels) != 0:
                print(c)
                break


if __name__ == '__main__':
    retriever = FreebaseRetriever()
    grailqa_entity_classes()
