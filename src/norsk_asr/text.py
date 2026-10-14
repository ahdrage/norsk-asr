"""Text normalization for the "raw" target: lowercase, no punctuation, numbers as words.

`normalize` is used for both training targets and WER scoring, so hypotheses from every
model (ours and NB-Whisper) go through the same function as the references.
"""

import re
import unicodedata

from num2words import num2words

LETTERS = set("abcdefghijklmnopqrstuvwxyzæøåéèêóòôüäöáàç")

# Written abbreviations from Stortinget's edited transcripts, mapped to how they are spoken.
ABBREVIATIONS = {
    "pst.": "prosent",
    "kr.": "kroner",
    "mill.": "millioner",
    "mrd.": "milliarder",
    "nr.": "nummer",
    "bl.a.": "blant annet",
    "f.eks.": "for eksempel",
    "dvs.": "det vil si",
    "osv.": "og så videre",
    "ca.": "cirka",
    "evt.": "eventuelt",
    "jf.": "jamfør",
    "mv.": "med videre",
    "m.m.": "med mer",
    "o.l.": "og lignende",
    "pga.": "på grunn av",
    "iht.": "i henhold til",
    "ifm.": "i forbindelse med",
    "km": "kilometer",
}
SYMBOLS = {"%": " prosent ", "§": " paragraf ", "&": " og ", "+": " pluss ", "°": " grader "}

_abbr_re = re.compile(
    r"(?<![\w.])(" + "|".join(re.escape(a) for a in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")(?!\w)"
)
_thousands_re = re.compile(r"(?<=\d)[  .](?=\d{3}(?!\d))")
_number_re = re.compile(r"\d+(?:,\d+)?")


def _number_words(match: re.Match) -> str:
    s = match.group(0)
    if "," in s:
        whole, frac = s.split(",")
        return f" {num2words(int(whole), lang='no')} komma {' '.join(num2words(int(d), lang='no') for d in frac)} "
    return f" {num2words(int(s), lang='no')} "


def has_digits(text: str) -> bool:
    return any(c.isdigit() for c in text)


def normalize(text: str, numbers: bool = True) -> str:
    """Return the raw target form of `text`.

    With `numbers=False` digits are left in place, so callers can detect and drop them.
    Number expansion is a plain cardinal reading ("2025" -> "to tusen og tjuefem"), which
    will not match how years are spoken; training drops utterances with digits instead.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _abbr_re.sub(lambda m: f" {ABBREVIATIONS[m.group(1)]} ", text)
    for sym, word in SYMBOLS.items():
        text = text.replace(sym, word)
    if numbers:
        text = _thousands_re.sub("", text)
        text = _number_re.sub(_number_words, text)
    # Hyphens and slashes join words that are spoken separately ("co2-kompensasjon").
    text = re.sub(r"[-–—/]", " ", text)
    text = "".join(c if c in LETTERS or c.isdigit() or c.isspace() else " " for c in text)
    return " ".join(text.split())


def is_clean(normalized: str) -> bool:
    """True if the normalized text only contains letters we model and single spaces."""
    return bool(normalized) and all(c in LETTERS or c == " " for c in normalized)
