import logging
import os
import re
import sys
import time
from collections import Counter

import nltk
import pandas as pd
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import log_step


# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE       = os.path.join(BASE_DIR, 'data', 'output', 'reviews_cleaned.csv')
OUTPUT_FILE      = os.path.join(BASE_DIR, 'data', 'output', 'reviews_translated.csv')
TOP_WORDS_FILE   = os.path.join(BASE_DIR, 'data', 'input',  'top_words_raw.csv')
SLANG_DICT_FILE  = os.path.join(BASE_DIR, 'data', 'input',  'manado_slang_dict.json')

TOP_N_RAW = 300  # Number of top words to extract for dialect analysis


# ==========================================
# 2. TOP WORDS EXTRACTION (for dialect analysis)
# ==========================================
_GEMINI_SLANG_PROMPT = """
==============================================================
  COPY-PASTE PROMPT FOR GEMINI AI — MANADO SLANG DICTIONARY
==============================================================

Peran: Kamu adalah pakar linguistik dialek Manado (Sulawesi Utara) dan asisten penelitian NLP.

Konteks Data:
1. Saya sedang membangun pipeline NLP untuk menganalisis ulasan restoran di Google Maps dari wilayah Sulawesi Utara (Manado, Tomohon, Bitung).
2. Ulasan sudah melalui proses Google Translate (auto-detect → Bahasa Indonesia), namun kata-kata dialek Manado yang tidak dikenali oleh Google Translate kemungkinan masih tersisa dalam bentuk aslinya.
3. Saya akan mengunggah file bernama top_words_raw.csv yang berisi daftar {top_n} kata dengan frekuensi kemunculan tertinggi dari teks hasil terjemahan.

TUGASMU:
1. Identifikasi kata-kata dalam file tersebut yang merupakan dialek Manado, slang lokal Sulawesi Utara, atau singkatan informal yang TIDAK berhasil diterjemahkan oleh Google Translate.
2. Untuk setiap kata yang teridentifikasi, berikan padanan kata standar Bahasa Indonesia yang paling tepat.
3. Kembangkan kamus dengan menambahkan variasi penulisan umum dari kata yang sama (typo, singkatan, pengulangan huruf).

KRITERIA SELEKSI:
1. HANYA masukkan kata yang benar-benar bukan Bahasa Indonesia standar (dialek, slang, singkatan tidak baku).
2. JANGAN masukkan kata Bahasa Indonesia baku meskipun frekuensinya tinggi.
3. JANGAN masukkan nama tempat, nama orang, atau nama merek.
4. BOLEH menambahkan variasi penulisan yang tidak ada di file jika kamu yakin kata tersebut umum digunakan di konteks ulasan restoran Manado.

FORMAT OUTPUT (JSON siap pakai, tanpa komentar, tanpa penjelasan):
{{
  "slang_atau_singkatan": "kata_standar_bahasa_indonesia",
  "contoh_nda": "tidak",
  "contoh_torang": "kami"
}}

Jika kamu sudah mengerti instruksi ini, jawab dengan "Saya mengerti. Silakan unggah file top_words_raw.csv Anda."

==============================================================
"""


def extract_top_words_raw(translated_file, output_file, top_n=TOP_N_RAW):
    """
    Extracts the most frequent words from the translated review corpus (before
    dialect normalization and stemming). The output is used to identify
    Manado dialect words that survived Google Translate, which are then
    fed to Gemini AI to generate manado_slang_dict.json.

    Stopwords are filtered but negation words are preserved.
    Output format: CSV with columns [word, frequency] — same as top_words.csv.
    """
    logging.info(f"Extracting top {top_n} raw words from: {translated_file}")
    df = pd.read_csv(translated_file)
    df['review_text'] = df['review_text'].fillna('')

    nltk.download('stopwords', quiet=True)
    indonesian_stopwords = set(stopwords.words('indonesian'))
    negation_words = {'tidak', 'bukan', 'kurang', 'jangan', 'belum', 'tak', 'tiada', 'tanpa', 'enggan'}
    stopword_set = indonesian_stopwords - negation_words

    word_counter = Counter()
    for text in df['review_text']:
        if not isinstance(text, str) or not text.strip():
            continue
        words = text.split()
        word_counter.update(w for w in words if w not in stopword_set)

    top_words = word_counter.most_common(top_n)
    df_out = pd.DataFrame(top_words, columns=['word', 'frequency'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out.to_csv(output_file, index=False)
    logging.info(f"Top {top_n} raw words saved to: {output_file}")
    return top_n


def check_slang_dict(slang_dict_file, top_words_file, top_n):
    """
    Checks whether manado_slang_dict.json exists in pipeline/data/input/.
    If not, logs a copy-paste ready Gemini AI prompt and exits the pipeline.
    """
    if os.path.exists(slang_dict_file):
        return

    logging.warning("=" * 62)
    logging.warning("  MANADO SLANG DICTIONARY NOT FOUND")
    logging.warning("=" * 62)
    logging.warning(f"  Expected: {slang_dict_file}")
    logging.warning("")
    logging.warning("  Step 3 has extracted the top words from the translated corpus.")
    logging.warning(f"  Top words file: {top_words_file}")
    logging.warning("")
    logging.warning("  To generate the slang dictionary:")
    logging.warning("    1. Open the file top_words_raw.csv from pipeline/data/input/")
    logging.warning("    2. Feed it to Gemini AI using the prompt below")
    logging.warning("    3. Save Gemini's JSON output as: manado_slang_dict.json")
    logging.warning("    4. Place the file in: pipeline/data/input/")
    logging.warning("    5. Re-run the pipeline")
    logging.warning("")
    for line in _GEMINI_SLANG_PROMPT.format(top_n=top_n).strip().splitlines():
        logging.warning(f"  {line}")
    logging.warning("=" * 62)
    raise SystemExit(1)


# ==========================================
# 3. TRANSLATION
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

    # ---- Extract top words for Manado dialect analysis ----
    top_n = extract_top_words_raw(output_file, TOP_WORDS_FILE, TOP_N_RAW)

    # ---- Gate: stop pipeline if manado_slang_dict.json not yet generated ----
    check_slang_dict(SLANG_DICT_FILE, TOP_WORDS_FILE, top_n)

    log_step('Step 3 - Translation', f"{len(df):,} rows translated to Bahasa Indonesia | top_words_raw.csv exported ({TOP_N_RAW} words)")


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
