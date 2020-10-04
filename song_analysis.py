from dataclasses import dataclass
import csv
import math
import re

@dataclass
class Song:
    id: int
    title: str
    year: int
    artist: str
    genre: str
    lyrics: list

"""

Place your answers to the Design check questions here:

1. tf-idf is the product of the term frequency (tf) and document frequency (idf). 
tf-idf is a metric of a term's importance in a document with regards to a set of training data (a corpus), 
and in the context of this project, it represents the the importance of a word in a certain
song, related to the overall sample collection of songs. tf is calculated by counting the number of times
a word appears in a song, and idf measures the prevalence of a certain word in a set of lyrics by finding
the logarithm of the number of total songs divided by the number of songs containing that word. 

2. We believe that our tests need to be able to show: 1) our individual helper functions are correctly returning
the values of our requests and 2) our overall program returns accurate classification of genres. In terms of 1), 
we can feed the functions data and compared with visual inspection and manual calculation. For 2), we can father
actual classified songs from the Internet and compare the results between the published classification data
and the data our program generates (i.e. trying the real data set given in the assignment).

"""

bad_characters = re.compile(r"[^\w]")


def clean_word(word: str) -> str:
    """input: string
    output: string
    description: using the bad characters regular expression, this function strips out invalid
    characters
    """
    word = word.strip().lower()
    return bad_characters.sub("", word)


def clean_lyrics(lyrics: str) -> list:
    """input: string representing the lyrics for a song
    output: a list with each of the words for a song
    description: this function parses through all of the lyrics for a song and makes sure
    they contain valid characters
    """
    lyrics = lyrics.replace("\n", " ")
    return [clean_word(word) for word in lyrics.split(" ")]


def create_corpus(filename: str) -> list:
    """input: a filename
    output: a list of Songs
    description: this function is responsible for creating the collection of songs, including some data cleaning
    """
    with open(filename) as f:
        corpus = []
        iden = 0
        for s in csv.reader(f):
            if s[4] != "Not Available":
                new_song = Song(iden, s[1], s[2], s[3], s[4], clean_lyrics(s[5]))
                corpus.append(new_song)
                iden += 1
        return corpus


def compute_idf(corpus: list) -> dict:
    """input: a list of Songs
    output: a dictionary from words to inverse document frequencies (as floats)
    description: this function is responsible for calculating inverse document
      frequencies of every word in the corpus
    """
    # Create an empty word set and dictionary
    word_set = set()
    idf_dict = {}
    # Go through the corpus and add every lyric of each song to the set
    for l in corpus:
        word_set.update(l.lyrics)
        # Go through the set, make the initial idf value 0
    for word in word_set:
        word_idf_count = 0
        # go through the corpus once more, for every word in the set, check if the word exist in the lyrics of each song
        #if it does, increase its idf count
        #ince the program is doner going thru the corpus, calculate the idf values and input the in the dictionary
        for l in corpus:
            if word in l.lyrics:
                word_idf_count += 1
        idf_dict[word] = math.log((len(corpus) - 1) / word_idf_count)
    return idf_dict


def compute_tf(song_lyrics: list) -> dict:
    """input: list representing the song lyrics
    output: dictionary containing the term frequency for that set of lyrics
    description: this function calculates the term frequency for a set of lyrics"""
    # create an empty dictionary
    tf_dict = {}
    # Go through every word in the song lyrics list
    for ele in song_lyrics:
        # Make sure each word does not have invalid characters
        fix_word = clean_word(ele)
        # Check to see if the word is in dict
        # If it is, increase its count
        # if it is not, add it to the dictionary with a count of 1
        if fix_word in tf_dict.keys():
            tf_dict[fix_word] = tf_dict[fix_word] + 1
        else:
            tf_dict[fix_word] = 1
    return tf_dict


def compute_tf_idf(song_lyrics: list, corpus_idf: dict) -> dict:
    """input: a list representing the song lyrics and an inverse document frequency dictionary
    output: a dictionary with tf-idf weights for the song (words to weights)
    description: this function calculates the tf-idf weights for a song
    """
    # Utilize the tf fucntion to find the term frequency of the words in song_lyrics
    tf = compute_tf(song_lyrics)
    # create and empty dict
    tf_idf_dict = {}
    # go through each word in the lyrics
    for word in song_lyrics:
        # make sure the word does not have invalid charachters
        new_word = clean_word(word)
        # if the word exist in idf-dict, calcualte the tf-idf value of the word
        if new_word in corpus_idf.keys():
           tf_idf_dict[new_word] = tf[new_word] * corpus_idf[new_word]
        else:
            # if the word does not exist in corpus idf, just return the TF count
            tf_idf_dict[new_word] = tf[new_word]
    return tf_idf_dict


def compute_corpus_tf_idf(corpus: list, corpus_idf: dict) -> dict:
    """input: a list of songs and an idf dictionary
    output: a dictionary from song ids to tf-idf dictionaries
    description: calculates tf-idf weights for an entire corpus
    """
    # create and empty dict
    tf_idf_weights = {}
    for ele in corpus:
        # go through each element in corpus
        # add the tf-idf value of each song to the dict, witht the jeys being the ids of songs
        tf_idf_weights[ele.id] = compute_tf_idf(ele.lyrics, corpus_idf)
    return tf_idf_weights


def cosine_similarity(l1: dict, l2: dict) -> float:
    """input: dictionary containing the term frequency - inverse document frequency for a song,
    dictionary containing the term frequency - inverse document frequency for a song
    output: float representing the similarity between the values of the two dictionaries
    description: this function finds the similarity score between two dictionaries
    """
    magnitude1 = math.sqrt(sum(w * w for w in l1.values()))
    magnitude2 = math.sqrt(sum(w * w for w in l2.values()))
    dot = sum(l1[w] * l2.get(w, 0) for w in l1)
    return dot / (magnitude1 * magnitude2)


def nearest_neighbor(
    song_lyrics: str, corpus: list, corpus_tf_idf: dict, corpus_idf: dict
) -> Song:
    """input: a string representing the lyrics for a song, a list of songs,
      tf-idf weights for every song in the corpus, and idf weights for every word in the corpus
    output: a song object
    description: this function produces the song in the corpus that is most similar to the lyrics it is given
    """
    #make sure the lyrics of the song do not have invalid characters
    new_lyrics = clean_lyrics(song_lyrics)
    #make an empty dict
    score_table = {}
    for ele in corpus:
    # go trough eacfh element in corpus, calculate the cosine similarity of each song to the new song given
    # Add the cosine similarity value to the dict, the key being the id of the song that was used to comapre
        score_table[ele.id] = cosine_similarity(compute_tf_idf(new_lyrics, corpus_idf), corpus_tf_idf[ele.id])
    #Find the max value in the dictionary, and return the key
    max_id = max(score_table, key=lambda x: score_table[x])
    for ele in corpus:
        if ele.id == max_id:
            return ele


def main(filename: str, lyrics: str):
    corpus = create_corpus(filename)
    corpus_idf = compute_idf(corpus)
    corpus_tf_idf = compute_corpus_tf_idf(corpus, corpus_idf)
    print(nearest_neighbor(lyrics, corpus, corpus_tf_idf, corpus_idf).title)
