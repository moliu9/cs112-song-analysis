from song_analysis import *
import pytest
import math


# testing compute_idf
def testing_compute_idf():

    # the empty corpus
    assert compute_idf([]) == {}

    # two entries with no overlapping words: all idf values should be equal
    short_unique = create_corpus('short_unique.csv')
    assert compute_idf(short_unique) == {'love': math.log(2), 'lyrics': math.log(2),
                                         'baby': math.log(2), 'i': math.log(2),
                                         'birthday': math.log(2), 'you': math.log(2), 'happy': math.log(2)}

    # two entries with overlapping words: overlapped should have idf value of zero
    short_overlap = create_corpus('short_overlap.csv')
    assert compute_idf(short_overlap) == {'': math.log(2), 'to': math.log(2), 'lyrics': math.log(2),
                                          'love': math.log(2), 'i': math.log(2),
                                          'birthday': math.log(2), 'you': math.log(1), 'happy': math.log(2)}


def testing_compute_tf():
    pass


def testing_compute_tf_idf():
    pass


def testing_corpus_tf_idf():
    pass


def testing_nearest_neightbor():
    pass

