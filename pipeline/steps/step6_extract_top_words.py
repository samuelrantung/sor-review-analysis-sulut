import logging
import os
import sys
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import log_step


# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'output', 'reviews_preprocessed.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'output', 'top_words.csv')

# Number of top words to extract
TOP_N = 200

# Negation words excluded from stopword removal to preserve sentiment polarity
NEGATION_WORDS = {'tidak', 'bukan', 'kurang', 'jangan', 'belum', 'tak', 'tiada', 'tanpa', 'enggan'}


# ==========================================
# 2. WORD EXTRACTION
# ==========================================
def build_stopwords():
    """
    Builds the Indonesian stopword set from NLTK, excluding negation words
    that carry sentiment meaning and must be counted.
    """
    nltk.download('stopwords', quiet=True)  # No-op if already downloaded
    indonesian_stopwords = set(stopwords.words('indonesian'))
    return indonesian_stopwords - NEGATION_WORDS


def run_extract_top_words(input_file, output_file, top_n=TOP_N):
    """
    Extracts the most frequent words from the preprocessed review corpus.
    - Stopwords are filtered (negation words preserved)
    - Results are saved as a ranked CSV for manual review and lexicon enrichment
    """
    logging.info(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    df['review_text'] = df['review_text'].fillna('')
    logging.info(f"Loaded {len(df)} rows.")

    stopword_set = build_stopwords()

    logging.info(f"Extracting top {top_n} words from corpus...")
    word_counter = Counter()

    for text in df['review_text']:
        if not isinstance(text, str) or not text.strip():
            continue
        words = text.split()
        filtered_words = [word for word in words if word not in stopword_set]
        word_counter.update(filtered_words)

    total_unique = len(word_counter)
    top_words = word_counter.most_common(top_n)
    df_top_words = pd.DataFrame(top_words, columns=['word', 'frequency'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_top_words.to_csv(output_file, index=False)
    logging.info(f"Total unique words in corpus: {total_unique:,}")
    logging.info(f"Top {top_n} words saved to: {output_file}")
    log_step('Step 6 - Extract Top Words', f"Top {top_n} of {total_unique:,} unique words extracted from {len(df):,} rows")


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
            logging.FileHandler(os.path.join(BASE_DIR, 'data', 'output', 'step6_extract_top_words.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_extract_top_words(INPUT_FILE, OUTPUT_FILE)
