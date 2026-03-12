import streamlit as st
import pandas as pd
from copy import deepcopy
import random
import math
import plotly.graph_objects as go

from data import load_words, assign_probabilities
from solver import get_letter_scores, get_top_letters, get_top_words, get_pmf_data, foresee
from game import (
    initialize_game,
    apply_computer_guess,
    apply_solver_feedback,
    validate_solver_pattern,
    is_won,
    is_lost
)
from visuals import get_hangman_svg, build_pmf_figure

TITLE = "Entropy Hangman"
SUBTITLE = "An Information-Theoretic Hangman Solver"

st.set_page_config(page_title="Entropy Hangman", page_icon="hangman_favicon.svg", layout="wide")

KEYBOARD_ROWS = [
    list("qwertyuiop"),
    list("asdfghjkl"),
    list("zxcvbnm"),
]
HANGMAN_LETTER_FONT_SIZE = 32
HANGMAN_LETTER_SPACING = 6
MODE_OPTIONS = [
    "Play against computer (user guesses)",
    "Play against computer (computer guesses)",
    "Use as solver",
]
FREQ_LETTERS = [
    "e",
    "t",
    "a",
    "o",
    "i",
    "n",
    "s",
    "h",
    "r",
    "d",
    "l",
    "c",
    "u",
    "m",
    "w",
    "f",
    "g",
    "y",
    "p",
    "b",
    "v",
    "k",
    "j",
    "x",
    "q",
    "z"
]
SIMULATION_STRATEGY_OPTIONS = [
    "Entropy-optimal",
    "Random letters",
    "Most frequent letters",
]
SIMULATION_STRATEGY_TO_KEY = {
    "Entropy-optimal": "entropy",
    "Random letters": "random_until_singleton",
    "Most frequent letters": "freq_letters_until_singleton",
}

@st.cache_data
def cached_load_words():
    """Load dictionary data once"""
    return load_words()

@st.cache_data
def cached_assign_probabilities(dictionary: list[str], method: str):
    """Assign global word probabilities once per method"""
    return assign_probabilities(dictionary, method)

def reset_game(words_dict: dict[int, set[str]], freq_dict: dict[str, float], length: int, mode: str, freq_method: str) -> None:
    """Reset the current game in session state"""
    st.session_state.game = initialize_game(words_dict, freq_dict, length, mode, freq_method)
    st.session_state.selected_letter = None
    st.session_state.solver_pattern_input = "_" * length
    st.session_state.solver_pattern_text = "_" * length
    st.session_state.solver_history = []
    st.session_state.solver_toggle_guess = None
    st.session_state.solver_toggle_base_pattern = ""
    st.session_state.solver_toggle_positions = []

def mode_label_to_value(mode_label: str) -> str:
    """Map display labels to internal mode values"""
    if mode_label == "Play against computer (user guesses)":
        return "computer"
    if mode_label == "Play against computer (computer guesses)":
        return "computer_guesser"
    if mode_label == "Play against computer":
        return "computer"
    if mode_label == "Computer guesses your word":
        return "computer_guesser"
    return "solver"

def mode_value_to_label(mode: str) -> str:
    """Map internal mode values to display labels"""
    if mode == "computer":
        return "Play against computer (user guesses)"
    if mode == "computer_guesser":
        return "Play against computer (computer guesses)"
    return "Use as solver"

def render_letter_buttons(guessable_letters: set[str], key_prefix: str, disabled: bool = False) -> str | None:
    """Render a Wordle-like keyboard and return the clicked letter, if any"""
    clicked = None
    row_offsets = [0.0, 0.7, 1.7]

    for row_idx, row in enumerate(KEYBOARD_ROWS):
        if row_offsets[row_idx] == 0:
            cols = st.columns(len(row))
            start_idx = 0
        else:
            cols = st.columns([row_offsets[row_idx]] + [1] * len(row) + [row_offsets[row_idx]])
            start_idx = 1
        for col_idx, letter in enumerate(row):
            is_guessed = letter not in guessable_letters
            with cols[start_idx + col_idx]:
                if st.button(
                    letter.upper(),
                    key=f"{key_prefix}_{row_idx}_{col_idx}_{letter}",
                    use_container_width=True,
                    disabled=disabled or is_guessed
                ):
                    clicked = letter
    return clicked

def render_solver_pattern_selector(game: dict, solver_guess: str) -> str:
    """Render a clickable pattern builder and return the selected pattern string."""
    base_pattern = "".join(game["pattern"])
    needs_reset = (
        "solver_toggle_guess" not in st.session_state
        or "solver_toggle_base_pattern" not in st.session_state
        or "solver_toggle_positions" not in st.session_state
        or st.session_state.solver_toggle_guess != solver_guess
        or st.session_state.solver_toggle_base_pattern != base_pattern
        or len(st.session_state.solver_toggle_positions) != game["length"]
    )
    if needs_reset:
        st.session_state.solver_toggle_guess = solver_guess
        st.session_state.solver_toggle_base_pattern = base_pattern
        st.session_state.solver_toggle_positions = [False] * game["length"]

    st.write("Select underscore positions where this guessed letter appears:")

    for start_idx in range(0, game["length"], 10):
        cols = st.columns(10)
        for col_idx in range(10):
            i = start_idx + col_idx
            with cols[col_idx]:
                if i >= game["length"]:
                    st.write("")
                    continue

                current_char = game["pattern"][i]
                disabled = current_char != "_"
                if disabled:
                    label = current_char
                else:
                    label = solver_guess if st.session_state.solver_toggle_positions[i] else "_"

                if st.button(
                    label,
                    key=f"solver_pos_{i}",
                    use_container_width=True,
                    disabled=disabled
                ):
                    st.session_state.solver_toggle_positions[i] = not st.session_state.solver_toggle_positions[i]
                    st.rerun()

    new_pattern = []
    for i, current_char in enumerate(game["pattern"]):
        if current_char != "_":
            new_pattern.append(current_char)
        elif st.session_state.solver_toggle_positions[i]:
            new_pattern.append(solver_guess)
        else:
            new_pattern.append("_")
    return "".join(new_pattern)

def render_solver_feedback_controls(game: dict, solver_guess: str) -> None:
    """Render solver feedback controls for a guessed letter."""
    solver_pattern_input = render_solver_pattern_selector(game, solver_guess)
    st.markdown(
        f"<div style='font-size: {HANGMAN_LETTER_FONT_SIZE}px; letter-spacing: {HANGMAN_LETTER_SPACING}px; margin: 10px 0;'>{' '.join(list(solver_pattern_input))}</div>",
        unsafe_allow_html=True
    )

    apply_col, undo_col = st.columns([2, 1])
    with apply_col:
        apply_feedback_clicked = st.button("Apply solver feedback")
    with undo_col:
        undo_clicked = st.button(
            "Undo last feedback",
            disabled=len(st.session_state.solver_history) == 0,
            use_container_width=True
        )

    if undo_clicked and len(st.session_state.solver_history) > 0:
        st.session_state.game = st.session_state.solver_history.pop()
        st.rerun()

    if apply_feedback_clicked:
        ok, msg = validate_solver_pattern(solver_pattern_input, game["pattern"], solver_guess, game["length"])
        if not ok:
            st.error(msg)
        else:
            st.session_state.solver_history.append(deepcopy(game))
            apply_solver_feedback(game, solver_guess, solver_pattern_input.strip().lower().replace(" ", ""))
            st.rerun()

def select_best_guess(game: dict, freq_dict: dict[str, float]) -> str | None:
    """Return the best next letter guess for a game state."""
    if len(game["guessable_letters"]) == 0 or len(game["possible_words"]) == 0:
        return None

    if len(game["possible_words"]) == 1:
        only_word = next(iter(game["possible_words"]))
        for char in only_word:
            if char in game["guessable_letters"]:
                return char

    entropy_dict, _, hit_prob_dict = get_letter_scores(
        game["pattern"],
        game["possible_words"],
        game["guessable_letters"],
        freq_dict
    )
    top_guess = get_top_letters(entropy_dict, 1, hit_prob_dict=hit_prob_dict)
    if len(top_guess) == 0:
        return None
    return top_guess[0][0]

def select_simulation_guess(game: dict, freq_dict: dict[str, float], strategy: str) -> str | None:
    """Return a strategy-specific next guess for simulation."""
    if len(game["guessable_letters"]) == 0 or len(game["possible_words"]) == 0:
        return None

    # Once narrowed to one word, guess remaining letters from that word.
    if len(game["possible_words"]) == 1:
        only_word = next(iter(game["possible_words"]))
        for char in only_word:
            if char in game["guessable_letters"]:
                return char

    if strategy == "random_until_singleton":
        return random.choice(list(game["guessable_letters"]))

    if strategy == "freq_letters_until_singleton":
        for char in FREQ_LETTERS:
            if char in game["guessable_letters"]:
                return char
        return sorted(list(game["guessable_letters"]))[0]

    return select_best_guess(game, freq_dict)

def simulate_entropy_solver_games(
    words_dict: dict[int, set[str]],
    freq_dict: dict[str, float],
    length: int,
    freq_method: str,
    num_games: int,
    strategy: str
) -> tuple[dict[int, int], dict[int, list[str]]]:
    """Simulate games where computer both selects and guesses the word."""
    incorrect_distribution: dict[int, int] = {}
    examples_by_bin: dict[int, list[str]] = {}

    for _ in range(num_games):
        sim_game = initialize_game(words_dict, freq_dict, length, "computer", freq_method)
        target_word = sim_game["answer"]
        incorrect_guesses = 0
        # Keep simulating even after lives reach 0 so the incorrect-guess
        # distribution is not truncated at the loss threshold.
        while not is_won(sim_game) and len(sim_game["guessable_letters"]) > 0:
            guess = select_simulation_guess(sim_game, freq_dict, strategy)
            if guess is None:
                break

            pattern_before_guess = sim_game["pattern"].copy()
            possible_words_before_guess = sim_game["possible_words"]
            outcomes_for_guess = foresee(guess, pattern_before_guess, possible_words_before_guess)

            if guess not in sim_game["answer"]:
                incorrect_guesses += 1
            apply_computer_guess(sim_game, guess)
            pattern_after_guess = "".join(sim_game["pattern"])
            sim_game["possible_words"] = outcomes_for_guess.get(pattern_after_guess, set())

        # If a run somehow remains unsolved (e.g., no viable next guess),
        # still record its full incorrect-guess count.
        bin_value = incorrect_guesses
        incorrect_distribution[bin_value] = incorrect_distribution.get(bin_value, 0) + 1
        if bin_value not in examples_by_bin:
            examples_by_bin[bin_value] = []
        if target_word not in examples_by_bin[bin_value] and len(examples_by_bin[bin_value]) < 12:
            examples_by_bin[bin_value].append(target_word)

    return incorrect_distribution, examples_by_bin

def main():
    dictionary, words_dict = cached_load_words()

    st.title(TITLE)
    st.subheader(SUBTITLE)

    valid_lengths = sorted([length for length in words_dict.keys() if 3 <= length <= 20])

    left_col, right_col = st.columns([1, 1])

    if "game" not in st.session_state:
        initial_mode = "computer"
        initial_freq_method = "zipf"
        initial_length = valid_lengths[0]
        initial_freq_dict = cached_assign_probabilities(dictionary, initial_freq_method)
        reset_game(words_dict, initial_freq_dict, initial_length, initial_mode, initial_freq_method)

    game = st.session_state.game

    if "mode_radio" not in st.session_state:
        st.session_state.mode_radio = mode_value_to_label(game["mode"])
    elif st.session_state.mode_radio not in MODE_OPTIONS:
        st.session_state.mode_radio = mode_value_to_label(game["mode"])
    if "freq_method_select" not in st.session_state:
        st.session_state.freq_method_select = game["freq_method"]
    if "length_select" not in st.session_state:
        st.session_state.length_select = game["length"]
    if "pmf_letter_persist" not in st.session_state:
        st.session_state.pmf_letter_persist = "a"
    if "solver_history" not in st.session_state:
        st.session_state.solver_history = []
    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = None
    elif st.session_state.simulation_results is not None and "incorrect_distribution" not in st.session_state.simulation_results:
        st.session_state.simulation_results = None

    guesses_made = 26 - len(game["guessable_letters"])
    game_over = is_won(game) or is_lost(game)
    no_possible_words = len(game["possible_words"]) == 0
    no_dictionary_match = game["mode"] == "computer_guesser" and no_possible_words and not game_over
    settings_locked = guesses_made > 0 and not game_over and not no_dictionary_match

    if settings_locked:
        st.session_state.mode_radio = mode_value_to_label(game["mode"])
        st.session_state.freq_method_select = game["freq_method"]
        st.session_state.length_select = game["length"]

    with left_col:
        mode_col, reset_col = st.columns([2, 1])

        with mode_col:
            mode_label = st.radio(
                "Mode",
                MODE_OPTIONS,
                key="mode_radio",
                disabled=settings_locked
            )

        with reset_col:
            st.write("")
            st.write("")
            new_game_clicked = st.button("New game / reset", use_container_width=True)

    with right_col:
        settings_col1, settings_col2 = st.columns([1, 1])
        with settings_col1:
            st.selectbox(
                "Frequency method",
                ["uniform", "zipf"],
                key="freq_method_select",
                disabled=settings_locked
            )
        with settings_col2:
            st.selectbox(
                "Word length",
                valid_lengths,
                key="length_select",
                disabled=settings_locked
            )

    selected_mode = mode_label_to_value(st.session_state.mode_radio)
    selected_freq_method = st.session_state.freq_method_select
    selected_length = st.session_state.length_select

    if new_game_clicked:
        if "pmf_letter_select" in st.session_state:
            st.session_state.pmf_letter_persist = st.session_state.pmf_letter_select
        new_game_freq_dict = cached_assign_probabilities(dictionary, selected_freq_method)
        reset_game(words_dict, new_game_freq_dict, selected_length, selected_mode, selected_freq_method)
        st.rerun()

    if not settings_locked:
        settings_differ = (
            selected_mode != game["mode"]
            or selected_freq_method != game["freq_method"]
            or selected_length != game["length"]
        )
        if settings_differ:
            if "pmf_letter_select" in st.session_state:
                st.session_state.pmf_letter_persist = st.session_state.pmf_letter_select
            new_game_freq_dict = cached_assign_probabilities(dictionary, selected_freq_method)
            reset_game(words_dict, new_game_freq_dict, selected_length, selected_mode, selected_freq_method)
            st.rerun()

    game = st.session_state.game
    freq_dict = cached_assign_probabilities(dictionary, game["freq_method"])

    if (
        len(game["possible_words"]) > 1
        and len(game["guessable_letters"]) > 0
        and not is_won(game)
        and not is_lost(game)
    ):
        entropy_dict, outcomes_dicts_by_letter, hit_prob_dict = get_letter_scores(
            game["pattern"],
            game["possible_words"],
            game["guessable_letters"],
            freq_dict
        )
    else:
        entropy_dict, outcomes_dicts_by_letter, hit_prob_dict = {}, {}, {}

    with left_col:
        st.header("Game")

        hangman_col, wrong_col = st.columns([3, 2])

        with hangman_col:
            st.markdown(get_hangman_svg(game["lives"]), unsafe_allow_html=True)

        with wrong_col:
            wrong_letters = sorted(list(game["lacks_letters"]))
            wrong_letters_display = " ".join(wrong_letters) if len(wrong_letters) > 0 else "None"
            st.markdown("**Incorrect guesses**")
            st.markdown(
                f"<div style='font-size: {HANGMAN_LETTER_FONT_SIZE}px; letter-spacing: {HANGMAN_LETTER_SPACING}px; margin: 10px 0;'>{wrong_letters_display}</div>",
                unsafe_allow_html=True
            )


        st.markdown(
            f"<div style='font-size: {HANGMAN_LETTER_FONT_SIZE}px; letter-spacing: {HANGMAN_LETTER_SPACING}px; margin: 10px 0;'>{' '.join(game['pattern'])}</div>",
            unsafe_allow_html=True
        )

        st.write("Lives remaining:", game["lives"])

        if is_won(game):
            if game["mode"] == "computer_guesser":
                st.success(f"I found your word with {game['lives']} guess{'es' if game['lives'] != 1 else ''} remaining!")
            else:
                st.success(f"Congratulations, you found the word with {game['lives']} guess{'es' if game['lives'] != 1 else ''} remaining!")
        elif is_lost(game):
            if game["mode"] == "computer":
                st.error(f"You ran out of guesses! The word was: {game['answer']}")
            elif game["mode"] == "computer_guesser":
                st.error("I ran out of guesses! Congrats, you stumped me.")
            else:
                st.error("You ran out of guesses!")
        elif game["mode"] == "computer_guesser" and len(game["possible_words"]) == 0:
            st.error("No matching words remain for your feedback. Your word may not be in this dictionary or feedback may be inconsistent.")
            if st.button("Undo last feedback", disabled=len(st.session_state.solver_history) == 0, key="undo_no_match"):
                st.session_state.game = st.session_state.solver_history.pop()
                st.rerun()
        else:
            if game["mode"] == "computer":
                st.subheader("Guess a letter")
                clicked_guess = render_letter_buttons(game["guessable_letters"], "computer_guess")
                if clicked_guess is not None:
                    apply_computer_guess(game, clicked_guess)

                    if clicked_guess in outcomes_dicts_by_letter:
                        pattern_str = "".join(game["pattern"])
                        if pattern_str in outcomes_dicts_by_letter[clicked_guess]:
                            game["possible_words"] = outcomes_dicts_by_letter[clicked_guess][pattern_str]
                        else:
                            game["possible_words"] = set()

                    st.rerun()
            elif game["mode"] == "solver":
                st.subheader("Enter solver feedback")

                solver_guess = st.selectbox(
                    "Letter guessed",
                    sorted(list(game["guessable_letters"])) if len(game["guessable_letters"]) > 0 else [],
                    key="solver_guess_select"
                )

                render_solver_feedback_controls(game, solver_guess)
            else:
                st.subheader("Computer guess + your feedback")
                computer_guess = select_best_guess(game, freq_dict)
                if computer_guess is None:
                    st.warning("No letters left to guess.")
                else:
                    st.markdown(f"**Computer guesses:** `{computer_guess}`")
                    render_solver_feedback_controls(game, computer_guess)

        st.divider()
        st.header("Simulation")
        st.caption("See how different strategies stack up against each other in multiple simulated games.")

        sim_col1, sim_col2, sim_col3 = st.columns([2, 2, 1])
        with sim_col1:
            simulation_count = st.number_input(
                "Number of games",
                min_value=25,
                max_value=250,
                value=100,
                step=25,
                key="simulation_count"
            )
        with sim_col2:
            simulation_strategy_label = st.selectbox(
                "Simulation strategy",
                SIMULATION_STRATEGY_OPTIONS,
                index=0,
                key="simulation_strategy"
            )
        with sim_col3:
            st.write("")
            st.write("")
            run_simulation = st.button("Run simulation", use_container_width=True)

        if run_simulation:
            simulation_strategy_key = SIMULATION_STRATEGY_TO_KEY[simulation_strategy_label]
            with st.spinner("Running simulations..."):
                incorrect_distribution, examples_by_bin = simulate_entropy_solver_games(
                    words_dict,
                    freq_dict,
                    game["length"],
                    game["freq_method"],
                    int(simulation_count),
                    simulation_strategy_key
                )
            st.session_state.simulation_results = {
                "length": game["length"],
                "freq_method": game["freq_method"],
                "num_games": int(simulation_count),
                "strategy_label": simulation_strategy_label,
                "incorrect_distribution": incorrect_distribution,
                "examples_by_bin": examples_by_bin
            }

        if st.session_state.simulation_results is not None:
            sim_results = st.session_state.simulation_results
            st.write(
                f"Results for length {sim_results['length']} words, {sim_results['freq_method']} sampling, "
                f"{sim_results['num_games']} games, strategy: {sim_results.get('strategy_label', 'Entropy-optimal')}."
            )

            n_games = int(sim_results["num_games"])
            incorrect_distribution = sim_results["incorrect_distribution"]
            mean_incorrect = 0.0
            sem_mean = 0.0
            win_proportion = 0.0
            se_proportion = 0.0
            if n_games > 0:
                mean_incorrect = sum(bin_value * count for bin_value, count in incorrect_distribution.items()) / n_games
                if n_games > 1:
                    sample_var = (
                        sum(((bin_value - mean_incorrect) ** 2) * count for bin_value, count in incorrect_distribution.items())
                        / (n_games - 1)
                    )
                    sem_mean = math.sqrt(sample_var) / math.sqrt(n_games)

                n_wins = sum(count for bin_value, count in incorrect_distribution.items() if bin_value <= 5)
                win_proportion = n_wins / n_games
                se_proportion = math.sqrt((win_proportion * (1 - win_proportion)) / n_games)

            z_95 = 1.96
            mean_ci_low = mean_incorrect - z_95 * sem_mean
            mean_ci_high = mean_incorrect + z_95 * sem_mean
            p_ci_low = max(0.0, win_proportion - z_95 * se_proportion)
            p_ci_high = min(1.0, win_proportion + z_95 * se_proportion)

            st.write(
                f"Mean incorrect guesses: {mean_incorrect:.3f} ± {sem_mean:.3f} " 
                f"(95% confidence interval: [{mean_ci_low:.3f}, {mean_ci_high:.3f}])"
            )
            st.write(
                f"Win proportion: {win_proportion:.3f} ± {se_proportion:.3f} "
                f"(95% confidence interval: [{p_ci_low:.3f}, {p_ci_high:.3f}])"
            )

            filled_bins = sorted(
                [v for v, count in incorrect_distribution.items() if count > 0]
            )
            if len(filled_bins) == 0:
                filled_bins = [0]
            min_filled_bin = filled_bins[0]
            max_filled_bin = filled_bins[-1]

            chart_bins = list(range(min_filled_bin, max_filled_bin + 1))
            chart_labels = [str(v) for v in chart_bins]
            chart_values = [incorrect_distribution.get(v, 0) for v in chart_bins]
            hover_examples = []
            for bin_value in chart_bins:
                examples = sim_results["examples_by_bin"].get(bin_value, [])
                preview = ", ".join(examples[:8]) if len(examples) > 0 else "None"
                if len(examples) > 8:
                    preview += ", ..."
                hover_examples.append(preview)

            sim_fig = go.Figure()
            sim_fig.add_trace(go.Bar(
                x=chart_bins,
                y=chart_values,
                customdata=hover_examples,
                hovertemplate="Games: %{y}<br>Examples: %{customdata}<extra></extra>"
            ))
            show_loss_separator = max_filled_bin >= 6
            shapes = []
            if show_loss_separator:
                shapes.append(
                    dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=5.5,
                        x1=5.5,
                        y0=0,
                        y1=1,
                        line=dict(color="red", width=2)
                    )
                )

            sim_fig.update_layout(
                xaxis_title="Incorrect guesses",
                yaxis_title="Games",
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                shapes=shapes
            )
            axis_min = min_filled_bin - 0.5
            if show_loss_separator:
                axis_min = min(axis_min, 4.5)
            sim_fig.update_xaxes(
                fixedrange=True,
                tickmode="array",
                tickvals=chart_bins,
                ticktext=chart_labels,
                range=[axis_min, max_filled_bin + 0.5]
            )
            sim_fig.update_yaxes(fixedrange=True)
            st.plotly_chart(
                sim_fig,
                use_container_width=True,
                config={"scrollZoom": False}
            )

    with right_col:
        st.header("Solver")

        summary_col1, summary_col2 = st.columns([1, 1])

        with summary_col1:
            st.subheader("Top letter guesses")
            if len(entropy_dict) > 0:
                top_letters = get_top_letters(entropy_dict, 100, hit_prob_dict=hit_prob_dict)
                df_letters = pd.DataFrame(top_letters, columns=["Letter", "Entropy"])
                df_letters["Entropy"] = df_letters["Entropy"].map(lambda x: round(x, 3))
                st.dataframe(df_letters, hide_index=True, use_container_width=True, height=215)
            else:
                st.write("No letters to rank.")

        with summary_col2:
            st.subheader("Top possible words")
            if len(game["possible_words"]) > 0:
                top_words = get_top_words(game["possible_words"], freq_dict, 100)
                df_words = pd.DataFrame(top_words, columns=["Word", "Probability"])
                df_words["Probability"] = df_words["Probability"].map(lambda x: round(x, 6))
                st.dataframe(df_words, hide_index=True, use_container_width=True, height=215)
            else:
                st.write("No possible words remaining.")

        st.write("Number of possible words:", len(game["possible_words"]))

        if len(game["guessable_letters"]) > 0 and len(game["possible_words"]) > 0:
            pmf_letter_options = sorted(list(game["guessable_letters"]))
            if st.session_state.pmf_letter_persist in pmf_letter_options:
                desired_pmf_letter = st.session_state.pmf_letter_persist
            else:
                desired_pmf_letter = pmf_letter_options[0]

            if "pmf_letter_select" not in st.session_state or st.session_state.pmf_letter_select not in pmf_letter_options:
                st.session_state.pmf_letter_select = desired_pmf_letter

            selected_letter = st.selectbox(
                "Select a letter to inspect its post-guess PMF",
                pmf_letter_options,
                key="pmf_letter_select"
            )
            st.session_state.pmf_letter_persist = selected_letter

            pmf_data, entropy = get_pmf_data(selected_letter, game["pattern"], game["possible_words"], freq_dict)

            st.metric("Entropy", f"{entropy:.3f} bits")

            if len(pmf_data) > 0:
                fig = build_pmf_figure(pmf_data, selected_letter)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"scrollZoom": False}
                )

                pmf_table = pd.DataFrame([
                    {
                        "Pattern": item["pattern_spaced"],
                        "Probability": round(item["probability"], 6),
                        "Count": item["count"]
                    }
                    for item in pmf_data
                ])
                st.dataframe(pmf_table, hide_index=True, use_container_width=True)
            else:
                st.write("No PMF data available.")
        else:
            st.write("No solver visualization available.")

if __name__ == "__main__":
    main()
