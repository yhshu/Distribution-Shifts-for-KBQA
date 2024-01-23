# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from strsimpy import NormalizedLevenshtein
from strsimpy.jaro_winkler import JaroWinkler

jaro_winkler = JaroWinkler()
normalized_levenshtein = NormalizedLevenshtein()


def string_replace(string, mention, replacement):
    """
    Replaces a word in a string with another word.
    """
    pos = string.find(mention)
    if pos == -1:
        return string
    string = string.strip('? ')
    condition = 0  # start and end with space
    if ' ' in mention:
        condition = 3  # directly replace
    else:
        if pos == 0:
            condition = 1  # end with space
        elif pos + len(mention) == len(string):
            condition = 2  # start with space

    if condition == 0:
        string = string.replace(' ' + mention + ' ', ' ' + replacement + ' ')
        return string.replace(' ' + mention + ', ', ' ' + replacement + ', ')
    elif condition == 1:
        string = string.replace(mention + ' ', replacement + ' ')
        return string.replace(mention + ', ', replacement + ', ')
    elif condition == 2:
        return string.replace(' ' + mention, ' ' + replacement)
    elif condition == 3:
        return string.replace(mention, replacement)


def span_in_string(span, string):
    if span not in string:
        return False
    if ' ' in span and span in string:
        return True

    pos = string.find(span)
    string = string.strip('? ')
    if pos == 0:
        return (span + ' ' in string) or (span + ', ' in string) or (span + '. ' in string)
    elif pos + len(span) == len(string):
        return ' ' + span in string
    return (' ' + span + ' ' in string) or (' ' + span + ', ' in string) or (' ' + span + '. ' in string)


def dice_coefficient(a, b):
    """dice coefficient 2nt/(na + nb)."""
    if not len(a) or not len(b): return 0.0
    if len(a) == 1:  a = a + u'.'
    if len(b) == 1:  b = b + u'.'

    a_bigram_list = []
    for i in range(len(a) - 1):
        a_bigram_list.append(a[i:i + 2])
    b_bigram_list = []
    for i in range(len(b) - 1):
        b_bigram_list.append(b[i:i + 2])

    a_bigrams = set(a_bigram_list)
    b_bigrams = set(b_bigram_list)
    overlap = len(a_bigrams & b_bigrams)
    dice_coeff = overlap * 2.0 / (len(a_bigrams) + len(b_bigrams))
    return dice_coeff


def jaro_winkler_similarity(text_a: str, text_b: str):
    text_a = text_a.lower()
    text_b = text_b.lower()
    return jaro_winkler.similarity(text_a, text_b)


def normalized_levenshtein_similarity(text_a: str, text_b: str):
    return normalized_levenshtein.similarity(text_a, text_b)


def literal_similarity(text_a: str, text_b: str, substr_match=False):
    if substr_match is True:
        if text_a in text_b or text_b in text_a:
            return 1
    jaro_sim = jaro_winkler.similarity(text_a, text_b)
    norm_lev = normalized_levenshtein.similarity(text_a, text_b)
    dice_sim = dice_coefficient(text_a, text_b)
    res = (jaro_sim + norm_lev + dice_sim) / 3
    assert 0 <= res <= 1
    return res


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def is_subsequence(a, b):
    b = iter(b)
    return all(i in b for i in a)


def different_in_one_word(str1, str2):
    if str1.lower() == str2.lower():
        return True
    words1 = str1.split()
    words2 = str2.split()
    word_len_diff = abs(len(words1) - len(words2))
    if len(words1) == 1 or len(words2) == 1:
        return False
    if word_len_diff <= 1 and (str1.lower() in str2.lower() or str2.lower() in str1.lower()):
        return True
    if word_len_diff == 0:
        num_different_word = 0
        for i in range(0, len(words1)):
            if words1[i] != words2[i]:
                num_different_word += 1
                if num_different_word > 1:
                    return False
        return True
    return False


if __name__ == '__main__':
    pass
