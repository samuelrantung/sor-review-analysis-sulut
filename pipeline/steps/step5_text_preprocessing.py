import logging
import os
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import log_step


# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'output', 'reviews_normalized.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'output', 'reviews_preprocessed.csv')

# Negation words excluded from stopword removal to preserve sentiment polarity
NEGATION_WORDS = {'tidak', 'bukan', 'kurang', 'jangan', 'belum', 'tak', 'tiada', 'tanpa', 'enggan'}


# ==========================================
# 2. TEXT PREPROCESSING
# ==========================================
def build_stopwords():
    """
    Builds the Indonesian stopword set from NLTK, excluding negation words
    that carry sentiment meaning and must be preserved.
    """
    nltk.download('stopwords', quiet=True)  # No-op if already downloaded
    indonesian_stopwords = set(stopwords.words('indonesian'))
    return indonesian_stopwords - NEGATION_WORDS


def remove_stopwords(text, stopword_set):
    """Removes stopwords from a single text string, preserving negation words."""
    if not isinstance(text, str):
        return text
    words = text.split()
    return ' '.join(word for word in words if word not in stopword_set)


def run_preprocessing(input_file, output_file):
    """
    Performs stopword removal and stemming on review text:
    - Stopwords are removed using NLTK Indonesian stopword list
    - Negation words are preserved to maintain sentiment polarity
    - Stemming is applied using the Sastrawi Indonesian stemmer
    - Progress is logged every 100 rows during stemming
    """
    logging.info(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    logging.info(f"Loaded {total_rows} rows.")

    # Build stopword set
    stopword_set = build_stopwords()
    logging.info(f"Stopword set built: {len(stopword_set)} words (negation words excluded).")

    # Stopword removal
    logging.info("Applying stopword removal...")
    df['review_text'] = df['review_text'].apply(lambda x: remove_stopwords(x, stopword_set))

    # Drop reviews that became empty after stopword removal
    before = len(df)
    df = df[df['review_text'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
    dropped = before - len(df)
    if dropped > 0:
        logging.info(f"Dropped {dropped} rows that became empty after stopword removal.")
    logging.info("Stopword removal complete.")

    # Stemming using Sastrawi
    logging.info("Starting stemming process (this may take a while)...")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    stemmed_texts = []
    stemmed_count = 0
    skipped_count = 0
    for i, text in enumerate(df['review_text']):
        if isinstance(text, str):
            stemmed_texts.append(stemmer.stem(text))
            stemmed_count += 1
        else:
            stemmed_texts.append(text)
            skipped_count += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            logging.info(f"Stemming progress: {i + 1}/{len(df)} rows processed.")

    df['review_text'] = stemmed_texts
    logging.info("Stemming complete.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logging.info(f"Output saved to: {output_file}")
    log_step('Step 5 - Text Preprocessing', f"{total_rows:,} → {len(df):,} rows ({stemmed_count:,} stemmed, {skipped_count} skipped, {dropped} dropped empty)")


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
            logging.FileHandler(os.path.join(BASE_DIR, 'data', 'output', 'step5_text_preprocessing.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_preprocessing(INPUT_FILE, OUTPUT_FILE)
