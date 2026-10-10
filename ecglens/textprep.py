"""Report cleaning and language flagging for PTB-XL.

Facts verified against the HF mirror `longisland3/ptb-xl` (see README "Data notes"):
  * ptbxl_database.csv is valid UTF-8, but lowercase umlauts are stored as capitals
    (a -> A-umlaut, etc.); reading it as latin-1 makes this worse, not better.
  * ~1/3 of German reports end with the machine marker "4.46 <padding> unbestaetigter bericht".
  * ~1-2% of reports are Swedish; the German->English MarianMT model does not translate them.
"""
import re

import pandas as pd

_UMLAUT_FIX = str.maketrans({"Ä": "ä", "Ö": "ö", "Ü": "ü", "Å": "å"})
_MARKER = re.compile(r"\s*4\.46\s+unbest\S*\s+bericht\s*$", re.I)
_WS = re.compile(r"\s+")

_SWEDISH = re.compile(
    r"sinusrytm|vänster|höger|förmak|kammar|ospecifik|eljest|myokardaffektion|misstänkt|förändring|hjärt|blockering",
    re.I,
)
_ENGLISH = re.compile(
    r"\b(sinus rhythm|ecg|rate|axis|wave|waves|segment|segments|pathology|trace|abnormal|normal|infarct|"
    r"otherwise|consistent|possible|probable|the|and|of|is|are|with|no)\b",
    re.I,
)
_GERMAN = re.compile(
    r"sinusrhythmus|normales|lagetyp|linkstyp|rechtstyp|vorhof|infarkt|ekg|unspezifisch|kammer|schenkelblock|"
    r"extrasystole|st-|hypertrophie|ischämie|leitungs|niederspannung|tachykard|bradykard|bericht",
    re.I,
)


def clean_report(text: str) -> str:
    """Fix umlaut casing, strip the machine-generated marker, collapse whitespace, lowercase."""
    t = str(text).translate(_UMLAUT_FIX)
    t = _MARKER.sub("", t)
    return _WS.sub(" ", t).strip().lower()


def detect_lang(clean: str) -> str:
    """Heuristic language flag: 'sv' / 'en' / 'de'. Marker-based, ~exact for this templated corpus."""
    if _SWEDISH.search(clean):
        return "sv"
    # German check first for compound German words that contain English-looking tokens (e.g. 'ekg', 'st-')
    ger, eng = len(_GERMAN.findall(clean)), len(_ENGLISH.findall(clean))
    if ger > eng:
        return "de"
    return "en" if eng > 0 else "de"


def load_database(path: str) -> pd.DataFrame:
    """Read ptbxl_database.csv with the correct encoding and add report_clean / lang columns."""
    with open(path, "rb") as f:
        raw = f.read().decode("utf-8")
    from io import StringIO

    df = pd.read_csv(StringIO(raw), index_col="ecg_id")
    df["report_clean"] = df["report"].map(clean_report)
    df["lang"] = df["report_clean"].map(detect_lang)
    return df
