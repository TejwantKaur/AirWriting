from symspellpy import SymSpell, Verbosity

import pkg_resources

# SYMSPELL SETUP
sym_spell = SymSpell(
    max_dictionary_edit_distance=2,
    prefix_length=7
)

dictionary_path = (
    pkg_resources.resource_filename(
        "symspellpy",
        "frequency_dictionary_en_82_765.txt"
    )
)

sym_spell.load_dictionary(
    dictionary_path,
    term_index=0,
    count_index=1
)


# =====================================
# CHARACTER FIXES
# =====================================

CHAR_FIXES = {

    "0": "O",
    "1": "I",
    "3": "E",
    "4": "A",
    "5": "S",
    "8": "B",
}


# =====================================
# NORMALIZATION
# =====================================

def normalize_raw_word(raw_word):

    fixed = ""

    for char in raw_word.upper():

        fixed += CHAR_FIXES.get(
            char,
            char
        )

    return fixed


# =====================================
# BEST CORRECTION
# =====================================

def correct_word(raw_word):

    normalized = normalize_raw_word(
        raw_word
    )

    suggestions = sym_spell.lookup(
        normalized.lower(),
        Verbosity.CLOSEST,
        max_edit_distance=2,
    )

    if suggestions:

        return suggestions[0].term.upper()

    return normalized


# =====================================
# MULTIPLE SUGGESTIONS
# =====================================

def get_correction_suggestions(
    raw_word,
    limit=3
):

    normalized = normalize_raw_word(
        raw_word
    )

    suggestions = sym_spell.lookup(
        normalized.lower(),
        Verbosity.ALL,
        max_edit_distance=2,
    )

    words = []

    for suggestion in suggestions:

        word = suggestion.term.upper()

        if word not in words:

            words.append(word)

        if len(words) >= limit:
            break

    return words