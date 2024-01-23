# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import socket

hostname = socket.gethostname()
freebase_addr = 'localhost'
freebase_port = '3001'

grailqa_train_path = '../dataset/GrailQA/grailqa_v1.0_train.json'
grailqa_dev_path = '../dataset/GrailQA/grailqa_v1.0_dev.json'
grailqa_test_path = '../dataset/GrailQA/grailqa_v1.0_test_public.json'

grailqa_classes_path = '../dataset/GrailQA/grail_classes.txt'
grailqa_relations_path = '../dataset/GrailQA/grail_relations.txt'
grailqa_entity_classes_path = '../dataset/GrailQA/grailqa_entity_classes.txt'
fb_entity_classes_path = '../dataset/freebase_entity_classes.txt'
webqsp_ptrain_path = '../dataset/WebQSP/RnG/WebQSP.ptrain.expr.json'
webqsp_pdev_path = '../dataset/WebQSP/RnG/WebQSP.pdev.expr.json'
webqsp_test_path = '../dataset/WebQSP/RnG/WebQSP.test.expr.json'
graphq_ptrain_path = '../dataset/GraphQuestions/graph_questions_ptrain.json'
graphq_pdev_path = '../dataset/GraphQuestions/graph_questions_pdev.json'
graphq_train_path = '../dataset/GraphQuestions/graphquestions_v1_fb15_training_091420.json'
graphq_test_path = '../dataset/GraphQuestions/graphquestions_v1_fb15_test_091420.json'

# ontology
grailqa_reverse_properties_dict_path = '../dataset/GrailQA/ontology/reverse_properties'
grailqa_property_roles_dict_path = '../dataset/GrailQA/ontology/fb_roles'

sqb_train_path = '../dataset/SQB/fold-0.train.pickle'  # SimpleQuestions - Balance
sqb_dev_path = '../dataset/SQB/fold-0.valid.pickle'
sqb_test_path = '../dataset/SQB/fold-0.test.pickle'

# cache
freebase_cache_dir = '../fb_cache'
