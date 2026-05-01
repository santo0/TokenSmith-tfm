import re

from nltk.util import ngrams

# Regex for tokenizing KG / query text.
# Matches words (including hyphenated compounds and trailing '+').
KW_PATTERN = r"\b\w+(?:\s*-\s*\w+)*\+?"

# Simpler pattern for heading text (no hyphen compounds or '+' needed).
HEADING_PATTERN = r"\b\w+\b"


def extract_ngrams(text: str, pattern: str) -> set[str]:
    """Tokenize *text*, build unigrams + bigrams + trigrams, return as a set.

    Args:
        text:       Input string to tokenize.
        pattern:    Regex pattern used to extract tokens (e.g. ``KW_PATTERN``).

    Returns:
        Set of all n-gram strings (n = 1, 2, 3).
    """
    tokens = re.findall(pattern, text)
    all_terms = list(tokens)
    for n in (2, 3):
        all_terms.extend(" ".join(gram) for gram in ngrams(tokens, n))
    return set(all_terms)


def extract_ngrams_with_spans(
    text: str, pattern: str
) -> list[tuple[str, frozenset[int]]]:
    """Like extract_ngrams but tags each n-gram with its token-position span.

    Args:
        text:       Input string to tokenize.
        pattern:    Regex pattern used to extract tokens (e.g. ``KW_PATTERN``).

    Returns:
        List of ``(ngram_text, frozenset_of_token_indices)`` in token-sequence
        order. Unigrams, bigrams, and trigrams are all included.
    """
    tokens = re.findall(pattern, text)
    result: list[tuple[str, frozenset[int]]] = []
    for i, t1 in enumerate(tokens):
        result.append((t1, frozenset([i])))
        if i + 1 < len(tokens):
            result.append((f"{t1} {tokens[i + 1]}", frozenset([i, i + 1])))
        if i + 2 < len(tokens):
            result.append((
                f"{t1} {tokens[i + 1]} {tokens[i + 2]}",
                frozenset([i, i + 1, i + 2]),
            ))
    return result
