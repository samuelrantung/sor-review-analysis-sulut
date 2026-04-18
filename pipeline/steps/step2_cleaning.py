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

INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'input', 'all_reviews_raw.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'output', 'reviews_cleaned.csv')


# ==========================================
# 2. CLEANING
# ==========================================
def is_valid_rating(x):
    """Returns True if x is a numeric value within the valid Google Maps rating range (1-5)."""
    try:
        return 1 <= int(float(x)) <= 5
    except (ValueError, TypeError):
        return False


def run_cleaning(input_file, output_file):
    """
    Performs basic cleaning on raw Google Maps review data:
    - Validates required columns are present
    - Removes rows with missing review text or invalid rating
    - Normalizes and trims whitespace
    - Filters out reviews shorter than 3 words
    - Removes duplicate reviews
    - Resets DataFrame index before saving
    """
    logging.info(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    initial_count = len(df)
    logging.info(f"Loaded {initial_count} rows.")

    # Validate required columns exist before processing
    required_cols = ['review_text', 'review_rating', 'review_author', 'place_name']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logging.error(f"Missing required columns: {missing}. Aborting.")
        sys.exit(1)

    # Drop rows with missing review text
    df = df.dropna(subset=['review_text'])
    logging.info(f"After dropping null review_text  : {len(df)} rows ({initial_count - len(df)} removed)")

    # Drop rows with missing or invalid review rating (valid range: 1-5)
    before = len(df)
    df = df[df['review_rating'].apply(is_valid_rating)]
    logging.info(f"After dropping invalid rating    : {len(df)} rows ({before - len(df)} removed)")

    # Normalize newlines and carriage returns to a single space
    df['review_text'] = df['review_text'].apply(lambda x: re.sub(r'[\n\r]+', ' ', str(x)))

    # Trim extra whitespace before word count filter
    df['review_text'] = df['review_text'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

    # Filter out reviews with fewer than 3 words
    before = len(df)
    df = df[df['review_text'].apply(lambda x: len(str(x).split()) >= 3)]
    logging.info(f"After word count filter (< 3)    : {len(df)} rows ({before - len(df)} removed)")

    # Remove duplicate reviews based on content, author, and place
    before = len(df)
    df = df.drop_duplicates(subset=['review_text', 'review_author', 'place_name'])
    logging.info(f"After removing duplicates        : {len(df)} rows ({before - len(df)} removed)")

    retained_pct = len(df) / initial_count * 100
    logging.info(f"Summary: {len(df)} / {initial_count} rows retained ({retained_pct:.1f}%)")
    log_step('Step 2 - Cleaning', f"{initial_count:,} → {len(df):,} rows retained ({retained_pct:.1f}%)")

    df = df.reset_index(drop=True)

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
            logging.FileHandler(os.path.join(BASE_DIR, 'data', 'output', 'step2_cleaning.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_cleaning(INPUT_FILE, OUTPUT_FILE)
