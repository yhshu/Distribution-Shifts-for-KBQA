# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

class Link:
    type: str  # 'entity', 'relation'
    uri: str  # e.g., Freebase mid; dbr, dbo, dbp
    name: str  # entity name, or rdf label
    mention: str  # mention in the NLQ
    score: float  # linking score
    perfect_match: bool  # for GrailQA
    source: str  # 'elq', 'grail_qa', 'falcon2'

    def __init__(self):
        pass

    def init(self, t: int, uri: str, score: float, mention: str, name=None, perfect_match=None, source=None):
        self.type = t
        self.uri = uri
        self.score = score
        self.mention = mention
        self.name = name
        self.perfect_match = perfect_match  # from GrailQA linking
        self.source = source
        return self


def get_link_score(link: Link):
    return link.score


def type_check(var):
    if isinstance(var, Link):
        return var.uri
    return var
