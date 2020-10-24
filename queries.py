# By Roi Solomon
import nltk
import os
import string
import math
import sys

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python queries.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = dict()

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and filename.endswith(".txt"):
            with open(path, "r", encoding='utf8') as file:
                corpus[filename] = file.read()

    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by converting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document.lower())
    full_doc = []
    for word in words:
        # Words from stopwords corpus are being filtered
        if word not in nltk.corpus.stopwords.words("english") and word not in string.punctuation:
            full_doc.append(word)
    return full_doc


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    appears = dict()

    for filename in documents:
        obser_words = set()

        for word in documents[filename]:
            if word not in obser_words:
                obser_words.add(word)
                try:
                    appears[word] += 1
                    # Smoothing
                except KeyError:
                    appears[word] = 1
            # TF-IDF calculation
    return {word: math.log(len(documents) / appears[word]) for word in appears}


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = dict()

    for filename in files:
        tf_idfs[filename] = 0
        for word in query:
            tf_idfs[filename] += files[filename].count(word) * idfs[word]

    return [key for key, value in sorted(tf_idfs.items(), key=lambda item: item[1], reverse=True)][:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ranking = []

    for sentence in sentences:
        sentence_vals = [sentence, 0, 0]

        for word in query:
            if word in sentences[sentence]:
                sentence_vals[1] += idfs[word]
                sentence_vals[2] += sentences[sentence].count(
                    word) / len(sentences[sentence])

        ranking.append(sentence_vals)

    return [sentence for sentence, ab, cd, in sorted(ranking, key=lambda item: (item[1], item[2]), reverse=True)][:n]


if __name__ == "__main__":
    main()
