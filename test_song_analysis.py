from song_analysis import *
import pytest

# testing compute_idf
def testing_compute_idf():

    # the empty corpus
    assert compute_idf([]) == {}

    # two entries with no overlapping words: all idf values should be equal
    short_unique = create_corpus('short_unique.csv')
    assert compute_idf(short_unique) == {'love': 0.6931471805599453, 'lyrics': 0.6931471805599453, 'baby': 0.6931471805599453, 'i': 0.6931471805599453,
                                         'birthday': 0.6931471805599453, 'you': 0.6931471805599453, 'happy': 0.6931471805599453}

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
    assert compute_tf(["Is", "this", "the", "real", "life?", "Or", "is", "this", "just", "fantasy"]) == {"is": 2, "this":2, "the":1, "real":1, "life":1, "or":1, "just":1, "fantasy":1}
    assert compute_tf([]) == {}
    assert compute_tf(["MakinG", "SurE", "Edge", "Cases!@#", "WOrk%", "Work?"]) == {"making":1,"sure":1, "edge":1, "cases":1, "work":2}

def testing_compute_tf_idf():
    corpus_idf = compute_idf(create_corpus("old_example.csv"))
    assert compute_tf_idf(["Is", "this", "the", "real", "life?", "Or", "is", "this", "just", "fantasy"], corpus_idf) == {'is': 3.8918202981106265, 'this': 3.8918202981106265, 'the': 1.252762968495368, 'real': 1.9459101490553132, 'life': 1.9459101490553132, 'or': 1.9459101490553132, 'just': 1.252762968495368, 'fantasy': 1.9459101490553132}
    assert compute_tf_idf([], corpus_idf) == {}
    assert compute_tf_idf(["Is", "this", "the", "real", "life?", "Or", "is", "this", "just", "fantasy", "lol"], compute_idf(create_corpus("old_example.csv"))) == {'is': 3.8918202981106265, 'this': 3.8918202981106265, 'the': 1.252762968495368, 'real': 1.9459101490553132, 'life': 1.9459101490553132, 'or': 1.9459101490553132, 'just': 1.252762968495368, 'fantasy': 1.9459101490553132, 'lol': 1}



def testing_corpus_tf_idf():
    corpus_idf = compute_idf(create_corpus("long_example.csv"))
    corpus = create_corpus("long_example.csv")
    assert compute_corpus_tf_idf(corpus, corpus_idf) == {0: {'lyrics': 1.0986122886681098},
    1: {'is': 2.1972245773362196,'this': 2.1972245773362196, 'the': 1.0986122886681098, 'real': 1.0986122886681098,
    'life': 1.0986122886681098, 'or': 1.0986122886681098, 'just': 0.4054651081081644, 'fantasy': 1.0986122886681098},
    2: {'i': 0.8109302162163288, 'got': 2.1972245773362196, 'a': 1.0986122886681098, 'warzone': 1.0986122886681098,
    'in': 0.4054651081081644, 'my': 1.0986122886681098, 'head': 1.0986122886681098, 'dont': 1.0986122886681098,
    'stand': 1.0986122886681098, 'too': 1.0986122886681098, 'close': 1.0986122886681098, 'to': 1.0986122886681098,
    'me': 1.0986122886681098, 'ptsd': 1.0986122886681098}, 3: {'i': 0.8109302162163288, 'was': 1.0986122886681098,
    'in': 0.4054651081081644, 'love': 1.0986122886681098, 'with': 1.0986122886681098, 'pete': 1.0986122886681098,
    'now': 1.0986122886681098, 'just': 0.4054651081081644, 'sing': 1.0986122886681098, 'songs': 1.0986122886681098}}
    assert compute_corpus_tf_idf([], corpus_idf) == {}



def testing_nearest_neightbor():
   corpus = create_corpus("old_example.csv")
   corpus_idf = compute_idf(create_corpus("old_example.csv"))
   corpus_tf_idf = compute_corpus_tf_idf(corpus, corpus_idf)
   assert nearest_neighbor("Todavia yo te quiero, pero se que es un error bebe."
                           " Tu ya no eres el amor mio",
   corpus,corpus_tf_idf, corpus_idf) == Song(id=7, title='Mia', year='2009',
     artist='Bad Bunny', genre='Pop-Latino',lyrics=
    ['bebe', 'tu', 'eres', 'mia', 'yo', 'te', 'amo', 'mi', 'amor','i',
     'love', 'you'])


