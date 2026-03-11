import math
from collections import defaultdict

def normalize_possible_word_probs(possible_words: set[str], freq_dict: dict[str, float]) -> dict[str, float]:
    """Normalize the word probabilities over the current possible words"""
    norm = 0.0
    for word in possible_words:
        norm += freq_dict[word]
    if norm == 0:
        return {word: 0.0 for word in possible_words}
    return {word: freq_dict[word] / norm for word in possible_words}

def foresee(letter: str, pattern: list[str], possible_words: set[str]) -> dict[str, set[str]]:
    """Based on a guessed letter and possible words, return a dict: pattern | {words}"""
    outcomes_dict = defaultdict(set)
    for word in possible_words:
        new_pattern = pattern.copy()
        for i, char in enumerate(word):
            if char == letter:
                new_pattern[i] = letter
        outcomes_dict["".join(new_pattern)].add(word)
    return outcomes_dict

def calculate_entropy(
    possible_words: set[str],
    outcomes_dict: dict[str, set[str]],
    freq_dict: dict[str, float],
    norm: float | None = None
) -> float:
    """Calculate entropy of an outcome distribution using word frequencies"""
    if norm is None:
        norm = 0.0
        for word in possible_words:
            norm += freq_dict[word]
    if norm == 0:
        return 0.0

    entropy = 0.0
    for outcome_pattern in outcomes_dict:
        p = 0.0
        for word in outcomes_dict[outcome_pattern]:
            p += freq_dict[word]
        p /= norm
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def get_letter_scores(
    pattern: list[str],
    possible_words: set[str],
    guessable_letters: set[str],
    freq_dict: dict[str, float]
) -> tuple[dict[str, float], dict[str, dict[str, set[str]]], dict[str, float]]:
    """Return entropy and hit-probability by letter, along with all outcome dicts."""
    entropy_dict = {}
    outcomes_dicts_by_letter = {}
    hit_prob_dict = {}
    norm = 0.0
    for word in possible_words:
        norm += freq_dict[word]

    pattern_str = "".join(pattern)
    for letter in guessable_letters:
        outcomes_dict = foresee(letter, pattern, possible_words)
        outcomes_dicts_by_letter[letter] = outcomes_dict
        entropy_dict[letter] = calculate_entropy(possible_words, outcomes_dict, freq_dict, norm=norm)
        p_miss = 0.0
        for word in outcomes_dict.get(pattern_str, set()):
            p_miss += freq_dict[word]
        if norm > 0:
            p_miss /= norm
        else:
            p_miss = 1.0
        hit_prob_dict[letter] = 1.0 - p_miss
    return entropy_dict, outcomes_dicts_by_letter, hit_prob_dict

def get_top_letters(
    entropy_dict: dict[str, float],
    k: int = 5,
    hit_prob_dict: dict[str, float] | None = None
) -> list[tuple[str, float]]:
    """Return the top-k letters by entropy, tie-broken by hit probability."""
    if hit_prob_dict is None:
        hit_prob_dict = {}
    entropy_dict_items = sorted(
        entropy_dict.items(),
        key=lambda item: (-round(item[1], 12), -hit_prob_dict.get(item[0], 0.0), item[0])
    )
    return entropy_dict_items[:k]

def get_top_words(possible_words: set[str], freq_dict: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    """Return the top-k most likely words among the remaining possibilities"""
    probs = normalize_possible_word_probs(possible_words, freq_dict)
    items = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
    return items[:k]

def get_pmf_data(letter: str, pattern: list[str], possible_words: set[str], freq_dict: dict[str, float]) -> tuple[list[dict], float]:
    """Return PMF data for a letter guess, along with its entropy"""
    outcomes_dict = foresee(letter, pattern, possible_words)
    norm = 0.0
    for word in possible_words:
        norm += freq_dict[word]

    pmf_data = []
    for outcome_pattern in outcomes_dict:
        p = 0.0
        words_here = outcomes_dict[outcome_pattern]
        for word in words_here:
            p += freq_dict[word]
        p /= norm

        words_sorted_by_freq = sorted(words_here, key=lambda word: (-freq_dict[word], word))

        pmf_data.append({
            "pattern": outcome_pattern,
            "pattern_spaced": " ".join(list(outcome_pattern)),
            "probability": p,
            "count": len(words_here),
            "words": words_sorted_by_freq
        })

    pmf_data.sort(key=lambda item: (-item["probability"], item["pattern"]))
    entropy = calculate_entropy(possible_words, outcomes_dict, freq_dict, norm=norm)
    return pmf_data, entropy
