import logging
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import log_step


# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'output', 'reviews_translated.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'output', 'reviews_normalized.csv')

MANADO_SLANG_DICT = {
    # Negation & Time
    'nda': 'tidak', 'nyanda': 'tidak', 'nyd': 'tidak', 'nynd': 'tidak',
    'bukang': 'bukan', 'bkn': 'bukan', 'blm': 'belum', 'blum': 'belum',
    'so': 'sudah', 'smo': 'sudah mau', 'ja': 'sering', 'ga': 'tidak',
    'gak': 'tidak', 'ngga': 'tidak', 'nggak': 'tidak', 'tdk': 'tidak',
    'nd': 'tidak', 'udah': 'sudah', 'mo': 'mau',

    # Pronouns & Possession
    'tape': 'saya punya', 'dpe': 'nya', 'depe': 'nya', 'p': 'punya',
    'torang': 'kami', 'trg': 'kami', 'dorang': 'mereka', 'drg': 'mereka',
    'ngoni': 'kalian', 'qt': 'saya', 'pa': 'pada',

    # Verbs / Actions
    'bekeng': 'membuat', 'bking': 'membuat', 'bkg': 'membuat', 'bking2': 'membuat',
    'kase': 'beri', 'kse': 'beri', 'makang': 'makan', 'mkg': 'makan', 'mkng': 'makan',
    'ambe': 'ambil', 'bale': 'kembali', 'pi': 'pergi', 'lia': 'lihat',
    'dtg': 'datang',

    # Adjectives, Adverbs & Conjunctions
    'sadap': 'enak', 'sdaaaap': 'enak', 'skli': 'sekali', 'sx': 'sekali', 'skalianan': 'sekali',
    'kong': 'lalu', 'mar': 'tapi', 'pe': 'sangat', 'jo': 'saja', 'leh': 'juga',
    'akang': 'nya', 'akg': 'nya', 'katu': 'ternyata', 'tu': 'itu', 'tuu': 'itu',
    'laeng': 'lain', 'tampa': 'tempat', 'tmpt': 'tempat', 'dg': 'dengan', 'dnk': 'dengan',
    'ujang': 'hujan', 'capat': 'cepat', 'biongo': 'bingung', 'bingo': 'bingung',
    'itang': 'hitam', 'basar': 'besar', 'cma': 'cuma', 'tllu': 'terlalu', 'kiapa': 'kenapa'
}


# ==========================================
# 2. NORMALIZATION
# ==========================================
# Compile all dictionary keys into a single regex pattern for efficiency
_PATTERN = re.compile(
    r'\b(' + '|'.join(map(re.escape, MANADO_SLANG_DICT.keys())) + r')\b',
    re.IGNORECASE
)


def normalize_dialect(text):
    """
    Normalizes Manado dialect slang and common abbreviations to standard Bahasa Indonesia.
    Also replaces 'salah satu' with 'satu' to prevent the word 'salah' from being
    incorrectly detected as a negative sentiment token in the lexicon.
    Returns a tuple of (normalized_text, substitution_count).
    """
    if not isinstance(text, str):
        return text, 0

    text = text.lower()
    text = re.sub(r'\bsalah satu\b', 'satu', text)
    normalized, count = _PATTERN.subn(
        lambda match: MANADO_SLANG_DICT[match.group(0).lower()], text
    )
    return normalized, count


def run_normalization(input_file, output_file):
    """
    Applies Manado dialect normalization to all review texts.
    Logs total substitutions made for quantitative reporting.
    """
    logging.info(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    logging.info(f"Loaded {len(df)} rows.")

    total_substitutions = 0
    rows_affected = 0
    normalized_texts = []

    for text in df['review_text']:
        result, count = normalize_dialect(text)
        normalized_texts.append(result)
        if count > 0:
            total_substitutions += count
            rows_affected += 1

    df['review_text'] = normalized_texts

    affected_pct = rows_affected / len(df) * 100
    logging.info(f"Normalization complete. {total_substitutions:,} substitutions made across {rows_affected} rows ({affected_pct:.1f}% of corpus).")
    log_step('Step 4 - Dialect Normalization', f"{total_substitutions:,} substitutions across {rows_affected} rows ({affected_pct:.1f}% of corpus)")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logging.info(f"Output saved to: {output_file}")


# ==========================================
# 3. ENTRY POINT
# ==========================================
if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'data', 'output'), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(BASE_DIR, 'data', 'output', 'step4_dialect_normalization.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_normalization(INPUT_FILE, OUTPUT_FILE)
