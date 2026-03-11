import pickle
from pathlib import Path
from collections import defaultdict
import math
import random

DICTIONARY = "dictionary.txt"
WORDS_DICT = "words_dict.pkl"
FREQ_METHOD = "zipf"
ALPHABET = set([chr(i) for i in range(ord('a'), ord('z') + 1)])

def load_words(force_reload: bool = False) -> tuple[list, dict[int, set[str]]]:
    """Using .txt and/or .pkl files, load the dict: length | {words}"""
    if Path(DICTIONARY).is_file():
        with open(DICTIONARY, "r") as f:
            dictionary = [line.rstrip() for line in f]
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

def assign_probabilities(dictionary: list, method: str) -> dict[str, float]:
    """Assign frequency probabilities to all words using either a uniform or Zipf distribution"""
    freq_dict = {}
    if method == "uniform":
        p = 1 / len(dictionary)
        for word in dictionary:
            freq_dict[word] = p
    elif method == "zipf":
        norm = 0
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
    
def pick_word(possible_words: set[str], freq_dict: dict, method: str) -> str:
    """Pick a random answer word from the list of possible words using word frequencies"""
    if method == "uniform":
        return random.choice(list(possible_words))
    elif method == "zipf":
        norm = 0
        for word in possible_words:
            norm += freq_dict[word]
        t = 0.0
        target = random.uniform(0, norm)
        for word in possible_words:
            t += freq_dict[word]
            if t > target:
                return word
        return word
    else:
        raise ValueError("Invalid frequency method.")

def foresee(letter: str, pattern: list, possible_words: set[str]) -> dict[str, set[str]]:
    """Based on a guessed letter and possible words, return a dict: pattern | {words}"""
    outcomes_dict = defaultdict(set)
    new_pattern = []
    for word in possible_words:
        new_pattern = pattern.copy()
        for i, char in enumerate(word):
            if char == letter:
                new_pattern[i] = letter
        outcomes_dict["".join(new_pattern)].add(word)
    return outcomes_dict

def calculate_entropy(possible_words: set, outcomes_dict: dict, freq_dict: dict) -> float:
    """Calculate entropy of an outcome distribution using word frequencies"""
    norm = 0.0
    for word in possible_words:
        norm += freq_dict[word]
    entropy = 0.0
    for pattern in outcomes_dict:
        p = 0.0
        for word in outcomes_dict[pattern]:
            p += freq_dict[word]
        p /= norm
        entropy -= p * math.log2(p)
    return entropy

def update_pattern(guess: str, answer: str, pattern: list) -> None:
    """Update the clue pattern with a new guess"""
    for i, char in enumerate(answer):
        if char == guess:
            pattern[i] = char

def main():
    dictionary, words_dict = load_words()
    freq_dict = assign_probabilities(dictionary, FREQ_METHOD)
    word_lengths = list(words_dict.keys())
    word_lengths.sort()
    lives = 6
    
    length = input("Select length from " + str(word_lengths) + ": ").strip()
    while True:
        try:
            length = int(length)
            if length in word_lengths:
                break
            else:
                length = input("Select length from " + str(word_lengths) + ": ").strip()
        except:
            length = input("Select length from " + str(word_lengths) + ": ").strip()

    possible_words = words_dict[length]
    answer = pick_word(possible_words, freq_dict, FREQ_METHOD)

    guessable_letters = ALPHABET.copy()
    lacks_letters = set()
    pattern = ["_"] * length

    while lives > 0 and any(pattern[i] == "_" for i in range(len(pattern))):
        entropy_dict = {}
        outcomes_dicts_by_letter = {}
        for letter in guessable_letters:
            outcomes_dict = foresee(letter, pattern, possible_words)
            outcomes_dicts_by_letter[letter] = outcomes_dict
            entropy_dict[letter] = round(calculate_entropy(possible_words, outcomes_dict, freq_dict), 3)

        entropy_dict_items = sorted(entropy_dict.items(), key=lambda item: item[1], reverse = True)
        entropy_dict_sorted = dict(entropy_dict_items)
        print(entropy_dict_sorted)
        print(" ".join(pattern))
        print(f"Nonexistent letters: {lacks_letters if len(lacks_letters) > 0 else ""}")
        print(f"Lives: {lives}")

        guess = input("Guess a letter: ").strip()
        while guess not in guessable_letters:
            guess = input("Guess a letter: ").strip()

        guessable_letters.remove(guess)
        if guess in answer:
            update_pattern(guess, answer, pattern)
        else:
            lives -= 1
            lacks_letters.add(guess)
        possible_words = outcomes_dicts_by_letter[guess]["".join(pattern)]
        print("")
    
    if lives == 0:
        print("You ran out of guesses! The word was:")
        print(" ".join(answer))
        print("Better luck next time!")
    else:
        print(" ".join(pattern))
        print(f"Congratulations, you found the word with {lives} guess{"" if lives == 1 else "es"} remaining!")
        
if __name__ == "__main__":
    main()