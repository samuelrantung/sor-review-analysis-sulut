import logging
import os
import re
import sys
import time

import pandas as pd
from deep_translator import GoogleTranslator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import log_step


# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE  = os.path.join(BASE_DIR, 'data', 'output', 'reviews_cleaned.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'output', 'reviews_translated.csv')


# ==========================================
# 2. TRANSLATION
# ==========================================
def run_translation(input_file, output_file):
    """
    Translates all review text to Bahasa Indonesia using Google Translate.
    - Supports resume: skips rows already marked as translated
    - Auto-saves progress every 50 rows to prevent data loss
    - Applies case folding and removes non-alphabetic characters after translation
    - Drops the internal 'is_translated' tracking column before saving final output
    """
    # Resume: load existing output if available, otherwise load input
    if os.path.exists(output_file):
        logging.info(f"Existing output found. Resuming from: {output_file}")
        df = pd.read_csv(output_file)
    else:
        logging.info(f"Loading input from: {input_file}")
        df = pd.read_csv(input_file)

    if 'is_translated' not in df.columns:
        df['is_translated'] = False

    total_rows = len(df)
    already_done = df['is_translated'].apply(lambda x: str(x).lower() == 'true').sum()
    logging.info(f"Total rows: {total_rows} | Already translated: {already_done} | Remaining: {total_rows - already_done}")

    translator = GoogleTranslator(source='auto', target='id')
    interrupted = False
    processed_this_session = 0

    try:
        for index, row in df.iterrows():
            # Resume support: robust string/bool check
            if str(row.get('is_translated', False)).lower() == 'true':
                continue

            text = row['review_text']

            if pd.isna(text) or not str(text).strip():
                df.at[index, 'is_translated'] = True
                processed_this_session += 1
                continue

            try:
                translated_text = translator.translate(str(text))

                # Guard: skip if translation returned None or empty string
                if not translated_text or not str(translated_text).strip():
                    logging.warning(f"Row {index}: translation returned empty result. Skipping.")
                    continue

                df.at[index, 'review_text'] = translated_text
                df.at[index, 'is_translated'] = True
                processed_this_session += 1
                time.sleep(0.3)

            except Exception as e:
                logging.warning(f"Row {index}: translation failed — {e}. Will retry on next run.")

            # Auto-save every 50 rows
            if processed_this_session > 0 and processed_this_session % 50 == 0:
                logging.info(f"Session progress: {processed_this_session} translated | Total done: {already_done + processed_this_session}/{total_rows}. Auto-saving...")
                df.to_csv(output_file, index=False)

    except KeyboardInterrupt:
        logging.warning("Translation interrupted by user.")
        interrupted = True
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        interrupted = True
    finally:
        # Always save progress with is_translated intact so resume works
        df.to_csv(output_file, index=False)
        logging.info(f"Progress saved. {processed_this_session} rows translated this session.")

    # Only apply post-processing and drop tracking column if fully complete
    if interrupted:
        logging.info("Translation incomplete. Re-run to continue from where it stopped.")
        return

    logging.info("All rows translated. Applying post-translation text processing...")
    df['review_text'] = df['review_text'].apply(
        lambda x: re.sub(r'\s+', ' ', re.sub(r'[^a-z\s]', '', str(x).lower())).strip()
    )

    # Drop internal tracking column before saving final output
    df = df.drop(columns=['is_translated'])

    df.to_csv(output_file, index=False)
    logging.info(f"Translation complete. {len(df)} rows saved to: {output_file}")
    log_step('Step 3 - Translation', f"{len(df):,} rows translated to Bahasa Indonesia")


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
            logging.FileHandler(os.path.join(BASE_DIR, 'data', 'output', 'step3_translation.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_translation(INPUT_FILE, OUTPUT_FILE)
