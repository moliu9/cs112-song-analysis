from song_analysis import *
import math
import pytest


# testing compute_idf
def testing_compute_idf():

    # the empty corpus
    assert compute_idf([]) == {}

    # two entries with no overlapping words: all idf values should be equal
    short_unique = create_corpus('short_unique.csv')
    assert compute_idf(short_unique) == {'love': math.log(2/1), 'lyrics': math.log(2/1), 'baby': math.log(2/1),
                                         'i': math.log(2/1),'birthday': math.log(2/1), 'you': math.log(2/1),
                                         'happy': math.log(2/1)}

    # two entries with overlapping words: overlapped should have idf value of zero
    short_overlap = create_corpus('short_overlap.csv')
    assert compute_idf(short_overlap) == {'': math.log(2/1), 'to': math.log(2/1), 'lyrics': math.log(2/1),
                                          'love': math.log(2/1), 'i': math.log(2/1), 'birthday': math.log(2/1),
                                          'you': 0.0, 'happy': math.log(2/1)}


def testing_compute_tf():

    # Empty list
    assert compute_tf([]) == {}

    # the empty string
    assert compute_tf(['']) == {'': 1}

    # Normal Scenario
    assert compute_tf(["Is", "this", "the", "real", "life?", "Or", "is","this", "just", "fantasy"]) == \
           {"is": 2, "this": 2, "the": 1, "real": 1, "life": 1, "or" :1, "just": 1, "fantasy": 1}

    # Making sure invalid characters are counted correctly
    assert compute_tf(["MakinG", "SurE", "Edge", "Cases!@#", "WOrk%", "Work?"])\
           == {"making": 1,"sure": 1, "edge": 1, "cases": 1, "work": 2}



def testing_compute_tf_idf():

    # empty word list and empty corpus
    assert compute_tf_idf([], {}) == {}

    # Normal scenario
    corpus_idf = compute_idf(create_corpus("old_example.csv"))
    assert compute_tf_idf(["Is", "this", "the", "real", "life?", "Or", "is", "this", "just", "fantasy"],
                          corpus_idf) == {'is': (math.log(7/1) * 2), 'this': (math.log(7/1) * 2),
                                          'the': (math.log(7/2) * 1), 'real': (math.log(7/1) * 1),
                                          'life': (math.log(7/1) * 1), 'or': (math.log(7/1) * 1),
                                          'just': (math.log(7/2) * 1), 'fantasy': (math.log(7/1) * 1)}

    # Test to see if a word in the lyric is not in the corpus
    assert compute_tf_idf(["Is", "this", "the", "real", "life?", "Or", "is", "this", "just", "fantasy", "lol"],
                          compute_idf(create_corpus("old_example.csv"))) == \
           {'is': (math.log(7/1) * 2), 'this': (math.log(7/1) * 2), 'the': (math.log(7/2) * 1),
            'real': (math.log(7/1) * 1), 'life': (math.log(7/1) * 1), 'or': (math.log(7/1) * 1),
            'just': (math.log(7/2) * 1), 'fantasy': (math.log(7/1) * 1), 'lol': 1}


def testing_corpus_tf_idf():

    # Testing an empty corpus
    assert compute_corpus_tf_idf([], {}) == {}

    # Testing if the definition produces the correct Tf-Idf values for the corpus
    corpus_idf = compute_idf(create_corpus("long_example.csv"))
    corpus = create_corpus("long_example.csv")

    assert compute_corpus_tf_idf(corpus, corpus_idf) == {0: {'lyrics': (math.log(3/1) * 1)},
                                                         1: {'is': (math.log(3/1) * 2),'this': (math.log(3/1) * 2),
                                                             'the': (math.log(3/1) * 1), 'real': (math.log(3/1) * 1),
                                                             'life': (math.log(3/1) * 1), 'or': (math.log(3/1) * 1),
                                                             'just': (math.log(3/2) * 1), 'fantasy': (math.log(3/1) * 1)},
                                                         2: {'i':(math.log(3/2) * 2),'got': (math.log(3/1) * 2),
                                                             'a': (math.log(3/1) * 1), 'warzone': (math.log(3/1) * 1),
                                                             'in': (math.log(3/2) * 1), 'my': (math.log(3/1) * 1),
                                                             'head': (math.log(3/1) * 1), 'dont':(math.log(3/1) * 1),
                                                             'stand': (math.log(3/1) * 1), 'too': (math.log(3/1) * 1),
                                                             'close': (math.log(3/1) * 1), 'to': (math.log(3/1) * 1),
                                                             'me':(math.log(3/1) * 1), 'ptsd': (math.log(3/1) * 1)},
                                                         3: {'i': (math.log(3/2) * 2), 'was': (math.log(3/1) * 1),
                                                             'in':(math.log(3/2) * 1), 'love': (math.log(3/1) * 1),
                                                             'with': (math.log(3/1) * 1), 'pete': (math.log(3/1) * 1),
                                                             'now': (math.log(3/1) * 1), 'just': (math.log(3/2) * 1),
                                                             'sing': (math.log(3/1) * 1), 'songs': (math.log(3/1) * 1)}}


def testing_nearest_neighbor():

   # Normal scenarios, test to see if function outputs songs that we determined would be most similar

   corpus = create_corpus("old_example.csv")
   corpus_idf = compute_idf(create_corpus("old_example.csv"))
   corpus_tf_idf = compute_corpus_tf_idf(corpus, corpus_idf)

   assert nearest_neighbor("Todavia yo te quiero, pero se que es un error bebe. Tu ya no eres el amor mio",
                           corpus, corpus_tf_idf, corpus_idf) == \
          Song(id=7, title='Mia', year='2009', artist='Bad Bunny', genre='Pop-Latino',
               lyrics= ['bebe', 'tu', 'eres', 'mia', 'yo', 'te', 'amo', 'mi', 'amor', 'i', 'love', 'you'])

   assert nearest_neighbor("Birthday", corpus, corpus_tf_idf, corpus_idf) == \
          Song(id=1, title='b-day', year='2009', artist='jesus Lopez', genre='Pop',
               lyrics=['happy', 'birthday', 'to', 'you', 'baby'])




