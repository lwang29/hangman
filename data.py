DICTIONARY = "dictionary.txt"
WORDS_DICT = "words_dict.pkl"
FREQ_METHOD = "zipf"
ALPHABET = set([chr(i) for i in range(ord('a'), ord('z') + 1)])

import pickle
from pathlib import Path
from collections import defaultdict

def load_words(force_reload: bool = False) -> tuple[list[str], dict[int, set[str]]]:
    """Using .txt and/or .pkl files, load the dict: length | {words}"""
    if Path(DICTIONARY).is_file():
        with open(DICTIONARY, "r") as f:
            dictionary = [line.rstrip().lower() for line in f if line.rstrip()]
    else:
        raise FileNotFoundError(f"{DICTIONARY} not found.")
    
    if not force_reload and Path(WORDS_DICT).is_file():
        with open(WORDS_DICT, "rb") as f:
            words_dict = pickle.load(f)
    else:
        words_dict = defaultdict(set)
        for s in dictionary:
            words_dict[len(s)].add(s)
        with open(WORDS_DICT, "wb") as f:
            pickle.dump(words_dict, f)

    return dictionary, words_dict

def assign_probabilities(dictionary: list[str], method: str) -> dict[str, float]:
    """Assign frequency probabilities to all words using either a uniform or Zipf distribution"""
    freq_dict = {}
    if method == "uniform":
        p = 1 / len(dictionary)
        for word in dictionary:
            freq_dict[word] = p
    elif method == "zipf":
        norm = 0.0
        invs = []
        for idx, word in enumerate(dictionary):
            p = 1 / (idx + 1)
            invs.append(p)
            norm += p
        for idx, word in enumerate(dictionary):
            freq_dict[word] = invs[idx] / norm
    else:
        raise ValueError("Invalid frequency method.")
    return freq_dict