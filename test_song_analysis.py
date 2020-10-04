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

    # the empty list
    assert compute_tf([]) == {}

    # the empty string
    assert compute_tf(['']) == {'': 1}

    # non-empty entries
    lyrics = clean_lyrics('happy birthday to you')
    assert compute_tf(lyrics) == {'happy': 1, 'birthday': 1, 'to': 1, 'you': 1}

    # non-empty, with numbers
    lyrics = clean_lyrics('h8ppy birthd3y to y0u')
    assert compute_tf(lyrics) == {'h8ppy': 1, 'birthd3y': 1, 'to': 1, 'y0u': 1}

    # non-empty, with bad characters
    lyrics = clean_lyrics('t[]his w^\old i,s on\ fir[e')
    assert compute_tf(lyrics) == {'this': 1, 'wold': 1, 'is': 1, 'on': 1, 'fire': 1}


def testing_compute_tf_idf():

    # empty word list and empty corpus
    assert compute_tf_idf([], {}) == {}

    # only unique words in lyrics
    idf = compute_idf(create_corpus('example.csv'))
    lyrics = clean_lyrics('close to love')
    assert compute_tf_idf(lyrics, idf) == {'close': math.log(7), 'to': math.log(7/3), 'love': math.log(7/2)}

    # repeating words in lyrics
    lyrics = clean_lyrics('this is now this is love love love')
    assert compute_tf_idf(lyrics, idf) == {'this': 2*math.log(7), 'is': 2*math.log(7),
                                           'now': math.log(7/2), 'love': 3*math.log(7/2)}


def testing_corpus_tf_idf():
    pass


def testing_nearest_neighbor():
    pass

