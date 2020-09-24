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

1.


2.


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
    pass


def compute_tf(song_lyrics: list) -> dict:
    """input: list representing the song lyrics
    output: dictionary containing the term frequency for that set of lyrics
    description: this function calculates the term frequency for a set of lyrics
    """
    pass


def compute_tf_idf(song_lyrics: list, corpus_idf: dict) -> dict:
    """input: a list representing the song lyrics and an inverse document frequency dictionary
    output: a dictionary with tf-idf weights for the song (words to weights)
    description: this function calculates the tf-idf weights for a song
    """
    pass


def compute_corpus_tf_idf(corpus: list, corpus_idf: dict) -> dict:
    """input: a list of songs and an idf dictionary
    output: a dictionary from song ids to tf-idf dictionaries
    description: calculates tf-idf weights for an entire corpus
    """
    pass


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
    pass


def main(filename: str, lyrics: str):
    corpus = create_corpus(filename)
    corpus_idf = compute_idf(corpus)
    corpus_tf_idf = compute_corpus_tf_idf(corpus, corpus_idf)
    print(nearest_neighbor(lyrics, corpus, corpus_tf_idf, corpus_idf).genre)