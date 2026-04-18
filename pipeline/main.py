import subprocess
import sys
import os

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
STEPS_DIR = os.path.join(BASE_DIR, 'steps')

# If this file already exists, Step 1 (scraping) is automatically skipped.
SCRAPING_OUTPUT = os.path.join(BASE_DIR, 'data', 'input', 'all_reviews_raw.csv')

steps = [
    ('Step 1 - Scraping',                'step1_scraping.py'),
    ('Step 2 - Cleaning',                'step2_cleaning.py'),
    ('Step 3 - Translation',             'step3_translation.py'),
    ('Step 4 - Dialect Normalization',   'step4_dialect_normalization.py'),
    ('Step 5 - Text Preprocessing',      'step5_text_preprocessing.py'),
    ('Step 6 - Extract Top Words',       'step6_extract_top_words.py'),
    ('Step 7 - Sentiment Scoring',       'step7_sentiment_scoring.py'),
    ('Step 8 - OLS Regression',           'step8_ols_regression.py'),
]


def run_step(name, filename):
    filepath = os.path.join(STEPS_DIR, filename)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, filepath], capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {name} failed. Pipeline stopped.")
        sys.exit(result.returncode)
    print(f"[OK] {name} done.")


if __name__ == '__main__':
    print("Starting pipeline...")

    for name, filename in steps:
        if filename == 'step1_scraping.py' and os.path.exists(SCRAPING_OUTPUT):
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            print(f"[SKIP] Raw data already exists: {SCRAPING_OUTPUT}")
            print(f"[SKIP] Step 1 skipped.")
            continue
        run_step(name, filename)

    print(f"\n{'='*60}")
    print("  Pipeline complete! Check results in pipeline/data/output/")
    print(f"{'='*60}\n")
