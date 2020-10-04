from song_analysis import *
import pytest


# testing compute_idf
def testing_compute_idf():

    # the empty corpus
    assert compute_idf([]) == {}

    # two entries with no overlapping words: all idf values should be equal
    short_unique = create_corpus('short_unique.csv')
    assert compute_idf(short_unique) == {'love': 0.6931, 'lyrics': 0.6931, 'baby': 0.6931, 'i': 0.6931,
                                         'birthday': 0.6931, 'you': 0.6931, 'happy': 0.6931}

    # two entries with overlapping words: overlapped should have idf value of zero
    # short_overlap = create_corpus('short_overlap.csv')
    # assert compute_idf(short_overlap) == {'': 0.6931,
    #                                       'to': 0.6931471805599453,
    #                                       'lyrics': 0.6931471805599453,
    #                                       'love': 0.6931471805599453,
    #                                       'i': 0.6931471805599453,
    #                                       'birthday': 0.6931471805599453,
    #                                       'you': 0.0,
    #                                       'happy': 0.6931471805599453}

    # three entries with same words appearing in multiple songs

def testing_compute_tf():
    pass


def testing_compute_tf_idf():
    pass


def testing_corpus_tf_idf():
    pass


def testing_nearest_neightbor():
    pass

