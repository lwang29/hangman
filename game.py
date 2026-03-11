import random
from data import ALPHABET

def pick_word(possible_words: set[str], freq_dict: dict[str, float], method: str) -> str:
    """Pick a random answer word from the list of possible words using word frequencies"""
    if method == "uniform":
        return random.choice(list(possible_words))
    elif method == "zipf":
        norm = 0.0
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

def initialize_game(words_dict: dict[int, set[str]], freq_dict: dict[str, float], length: int, mode: str, freq_method: str) -> dict:
    """Initialize a new hangman game state"""
    possible_words = set(words_dict[length])

    game = {
        "mode": mode,
        "freq_method": freq_method,
        "length": length,
        "lives": 6,
        "pattern": ["_"] * length,
        "guessable_letters": ALPHABET.copy(),
        "lacks_letters": set(),
        "possible_words": possible_words,
        "answer": None
    }

    if mode == "computer":
        game["answer"] = pick_word(possible_words, freq_dict, freq_method)

    return game

def update_pattern(guess: str, answer: str, pattern: list[str]) -> None:
    """Update the clue pattern with a new guess"""
    for i, char in enumerate(answer):
        if char == guess:
            pattern[i] = char

def apply_computer_guess(game: dict, guess: str) -> None:
    """Apply a guessed letter in computer-play mode"""
    if guess not in game["guessable_letters"]:
        return

    game["guessable_letters"].remove(guess)

    if guess in game["answer"]:
        update_pattern(guess, game["answer"], game["pattern"])
    else:
        game["lives"] -= 1
        game["lacks_letters"].add(guess)

def word_matches_state(word: str, pattern: list[str], lacks_letters: set[str]) -> bool:
    """Return whether a word is consistent with the current pattern and known absent letters"""
    for i, char in enumerate(pattern):
        if char != "_" and word[i] != char:
            return False
        if char == "_" and word[i] in set(pattern) - {"_"}:
            # if a revealed letter appears in a hidden slot, the pattern is inconsistent
            # for standard hangman where all copies are revealed at once
            return False

    for bad_char in lacks_letters:
        if bad_char in word:
            return False

    return True

def filter_possible_words(possible_words: set[str], pattern: list[str], lacks_letters: set[str]) -> set[str]:
    """Filter possible words by the visible pattern and known absent letters"""
    new_possible_words = set()
    for word in possible_words:
        if word_matches_state(word, pattern, lacks_letters):
            new_possible_words.add(word)
    return new_possible_words

def apply_solver_feedback(game: dict, guess: str, new_pattern_str: str) -> None:
    """Apply a guessed letter plus externally supplied feedback in solver mode"""
    if guess not in game["guessable_letters"]:
        return

    old_pattern = "".join(game["pattern"])
    new_pattern = list(new_pattern_str)

    game["guessable_letters"].remove(guess)

    if new_pattern_str == old_pattern:
        game["lacks_letters"].add(guess)
        game["lives"] -= 1

    game["pattern"] = new_pattern
    game["possible_words"] = filter_possible_words(game["possible_words"], game["pattern"], game["lacks_letters"])

def is_won(game: dict) -> bool:
    """Return whether the player has won"""
    return all(char != "_" for char in game["pattern"])

def is_lost(game: dict) -> bool:
    """Return whether the player has lost"""
    return game["lives"] <= 0

def validate_solver_pattern(pattern_str: str, old_pattern: list[str], guess: str, length: int) -> tuple[bool, str]:
    """Validate a solver-mode pattern entry"""
    pattern_str = pattern_str.strip().lower().replace(" ", "")
    if len(pattern_str) != length:
        return False, f"Pattern must have length {length}."
    for char in pattern_str:
        if char != "_" and not ("a" <= char <= "z"):
            return False, "Pattern can only contain underscores and lowercase letters."

    for i in range(length):
        if old_pattern[i] != "_" and pattern_str[i] != old_pattern[i]:
            return False, "You cannot remove letters that were already revealed."

    # if the guessed letter appears, it must be in all newly revealed places for that letter.
    # this validator keeps things simple and only ensures no obvious contradiction
    return True, ""