import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from report import log_step


# ==========================================
# 1. CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE            = os.path.join(BASE_DIR, 'data', 'output', 'reviews_preprocessed.csv')
OUTPUT_FILE           = os.path.join(BASE_DIR, 'data', 'output', 'reviews_scored.csv')
POS_LEXICON_FILE      = os.path.join(BASE_DIR, 'data', 'input',  'positive.tsv')
NEG_LEXICON_FILE      = os.path.join(BASE_DIR, 'data', 'input',  'negative.tsv')
SERVICE_KEYWORDS_FILE = os.path.join(BASE_DIR, 'data', 'input',  'service_keywords.txt')

# Negation words that flip the sentiment of the word that follows them.
# These are preserved from stopword removal in step5.
NEGATION_WORDS = {'tidak', 'bukan', 'kurang', 'jangan', 'belum', 'tak', 'tiada', 'tanpa'}


# ==========================================
# 2. LOADERS
# ==========================================
_GEMINI_PROMPT = """
==============================================================
  COPY-PASTE PROMPT FOR GEMINI AI
==============================================================

Peran: Kamu adalah pakar linguistik dan asisten penelitian untuk Aspect-Based Sentiment Analysis (ABSA) pada ulasan restoran platform Google Maps di Sulawesi Utara.
Konteks Data:
1. Saya sedang mengekstraksi Variabel X yaitu "Kualitas Layanan / Employee Behaviour" (Fokus HANYA pada interaksi manusia, staf, dan kecepatan layanan).
2. Teks ulasan berasal dari bahasa formal, informal, serta dialek lokal Manado yang sudah dinormalisasi.
3. Teks ulasan SUDAH MELALUI PROSES STEMMING (Sastrawi). Artinya, semua kata di dalam data saya murni berupa KATA DASAR tanpa awalan/akhiran.
TUGASMU: Saya akan mengunggah file bernama top_words.csv yang berisi daftar 200 kata dengan frekuensi kemunculan tertinggi dari data penelitian saya.
Tugasmu adalah menyeleksi dan mengekstraksi kata-kata dari file tersebut yang relevan dengan aspek "Kualitas Layanan / Employee Behaviour", lalu kembangkan menjadi daftar kamus kata kunci (Service Quality Lexicon).
Kriteria Pemilihan & Pengembangan Kata:
1. Mencakup aktor/subjek layanan (contoh: kasir, staf, koki, pelayan).
2. Mencakup atribut kinerja karyawan (contoh: ramah, cepat, lambat, lelet, senyum).
3. Mencakup kata dialek Manado terkait layanan (yang sudah dalam bentuk dasar/normalisasi).
SANGAT PENTING (BATASAN EXCLUSION):
1. WAJIB KATA DASAR: Semua kata yang kamu pilih dan kembangkan WAJIB dalam bentuk KATA DASAR (contoh: tulis "layan", jangan "pelayanan").
2. DILARANG memasukkan kata sentimen global atau evaluasi umum untuk mencegah tumpang tindih variabel kepuasan (seperti: mantap, puas, senang, rekomendasi, buruk, kecewa).
3. DILARANG memasukkan atribut fisik restoran atau makanan (seperti: nyaman, bersih, kotor, enak, jorok).
FORMAT OUTPUT: 
- Tanpa komentar, tanpa penomoran, tanpa duplikasi dari seed words
- Siap copy-paste langsung ke file .txt

Jika kamu sudah mengerti instruksi ini, jawab dengan "Saya mengerti. Silakan unggah file top_words.csv Anda."

==============================================================
"""


def load_service_keywords(filepath):
    """
    Loads the service quality keyword list from a plain text file (one word per line).
    This file is produced bottom-up from step6 top words, then expanded by Gemini AI.
    Returns a lowercase set for O(1) membership testing.

    If the file does not exist, logs a warning with a copy-paste ready Gemini AI prompt
    to help the user generate the file, then exits the pipeline.
    """
    if not os.path.exists(filepath):
        logging.warning("=" * 62)
        logging.warning("  SERVICE KEYWORDS FILE NOT FOUND")
        logging.warning("=" * 62)
        logging.warning(f"  Expected: {filepath}")
        logging.warning("")
        logging.warning("  Step 7 cannot run without this file.")
        logging.warning("  To generate it:")
        logging.warning("    1. Run Step 6 to produce top_words.csv")
        logging.warning("    2. Feed top_words.csv to Gemini AI using the prompt below")
        logging.warning("    3. Save Gemini's output as: service_keywords.txt")
        logging.warning("    4. Place the file in: pipeline/data/input/")
        logging.warning("")
        for line in _GEMINI_PROMPT.strip().splitlines():
            logging.warning(f"  {line}")
        logging.warning("=" * 62)
        raise SystemExit(1)

    with open(filepath, encoding='utf-8') as f:
        keywords = {line.strip().lower() for line in f if line.strip()}
    logging.info(f"Service keywords loaded: {len(keywords)} words from {filepath}")
    return keywords


def load_lexicon(pos_file, neg_file):
    """
    Loads the InSet sentiment lexicon from two TSV files (positive and negative).
    Returns a dict mapping word -> weight (positive weights > 0, negative < 0).
    """
    if not (os.path.exists(pos_file) and os.path.exists(neg_file)):
        raise FileNotFoundError(
            f"InSet lexicon files are required.\n"
            f"  Expected: {pos_file}\n"
            f"            {neg_file}"
        )

    pos_df = pd.read_csv(pos_file, sep='\t')
    neg_df = pd.read_csv(neg_file, sep='\t')

    lexicon = {}
    for _, row in pos_df.iterrows():
        lexicon[str(row['word']).strip().lower()] = float(row['weight'])
    for _, row in neg_df.iterrows():
        lexicon[str(row['word']).strip().lower()] = float(row['weight'])

    logging.info(f"Lexicon loaded: {len(pos_df)} positive words, {len(neg_df)} negative words.")
    return lexicon


# ==========================================
# 3. SCORING FUNCTIONS
# ==========================================
def calculate_sentiment_score(text, lexicon):
    """
    Computes a raw document-level sentiment score for a review using InSet lexicon weights.

    Applies negation scope: if a negation word (tidak, bukan, kurang, etc.) appears
    1 or 2 positions before a lexicon word, its weight is flipped (× -1).

    Returns a float: positive > 0, negative < 0, neutral = 0.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    words = text.split()
    score = 0.0

    for i, word in enumerate(words):
        weight = lexicon.get(word.lower(), 0.0)
        if weight == 0.0:
            continue
        negated = any(
            words[i - k].lower() in NEGATION_WORDS
            for k in (1, 2)
            if i - k >= 0
        )
        score += -weight if negated else weight

    return round(score, 4)


def classify_sentiment(score):
    """
    Converts a raw sentiment score to a categorical label.
    Returns 1 (Positive), -1 (Negative), or 0 (Neutral).
    """
    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0


ABSA_WINDOW_SIZE = 5  # Number of words to scan on each side of a matched keyword


def score_absa_service_quality(text, keyword_set, lexicon, window=ABSA_WINDOW_SIZE):
    """
    Performs Aspect-Based Sentiment Analysis (ABSA) for the service quality aspect.

    Algorithm:
      1. Tokenize the review text into words.
      2. Scan for any word that matches a service keyword (radar step).
      3. For each matched keyword, extract a context window of ±`window` words.
      4. Score sentiment of words within that window using the InSet lexicon,
         applying negation scope: if a negation word appears 1-2 positions before
         a lexicon word, its weight is flipped (multiplied by -1).
      5. Aggregate all window scores into a single aspect-level score.

    Returns:
       1  — service aspect is present and sentiment is positive
      -1  — service aspect is present and sentiment is negative
       0  — no service keyword found (aspect absent), or sentiment is neutral
    """
    if not isinstance(text, str) or not text.strip():
        return 0

    words = text.split()
    total_aspect_score = 0.0

    for i, word in enumerate(words):
        if word.lower() not in keyword_set:
            continue

        # Extract context window: ±window words around the matched keyword
        start = max(0, i - window)
        end   = min(len(words), i + window + 1)
        context = words[start:end]

        # Score words in the context window with negation scope handling.
        # If a negation word appears 1 or 2 positions before a lexicon word,
        # flip the weight of that lexicon word (× -1).
        window_score = 0.0
        for j, w in enumerate(context):
            weight = lexicon.get(w.lower(), 0.0)
            if weight == 0.0:
                continue
            # Check for negation word in the 1-2 positions immediately before
            negated = any(
                context[j - k].lower() in NEGATION_WORDS
                for k in (1, 2)
                if j - k >= 0
            )
            window_score += -weight if negated else weight

        total_aspect_score += window_score

    if total_aspect_score > 0:
        return 1
    elif total_aspect_score < 0:
        return -1
    else:
        return 0


# ==========================================
# 4. MAIN PIPELINE STEP
# ==========================================
def run_sentiment_scoring(input_file, output_file, pos_lexicon_file, neg_lexicon_file, service_keywords_file):
    """
    Scores each review for customer satisfaction (M) and service quality (X).

    Columns added:
      - satisfaction_raw_M   : raw float score (sum of InSet lexicon weights)
      - customer_satisfaction_M : categorical label: 1 (Positive), -1 (Negative), 0 (Neutral)
      - service_quality_X    : ABSA score: 1 (positive service), -1 (negative service), 0 (absent)

    These columns feed directly into the OLS regression model:
      Y (review_rating) ~ X (service_quality) + M (customer_satisfaction)
    """
    # ---- Load all inputs first (fail fast before any computation) ----
    logging.info(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    logging.info(f"Loaded {total_rows} rows.")

    # Validate required columns
    required_columns = {'review_text', 'review_rating'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    df['review_text'] = df['review_text'].fillna('')

    # Load all resources before computation begins
    lexicon = load_lexicon(pos_lexicon_file, neg_lexicon_file)
    keyword_set = load_service_keywords(service_keywords_file)

    # ---- Compute Variable M (Customer Satisfaction) ----
    logging.info("Computing customer satisfaction scores (Variable M)...")
    df['satisfaction_raw_M'] = df['review_text'].apply(
        lambda text: calculate_sentiment_score(text, lexicon)
    )
    df['customer_satisfaction_M'] = df['satisfaction_raw_M'].apply(classify_sentiment)

    counts = df['customer_satisfaction_M'].value_counts()
    pos_count = counts.get(1, 0)
    neg_count = counts.get(-1, 0)
    neu_count = counts.get(0, 0)
    logging.info(
        f"Customer satisfaction (M) distribution — Positive: {pos_count}, "
        f"Negative: {neg_count}, Neutral: {neu_count}"
    )

    # ---- Compute Variable X (ABSA Service Quality) ----
    logging.info(f"Computing ABSA service quality scores (Variable X, window=±{ABSA_WINDOW_SIZE})...")
    df['service_quality_X'] = df['review_text'].apply(
        lambda text: score_absa_service_quality(text, keyword_set, lexicon)
    )

    x_counts = df['service_quality_X'].value_counts()
    x_pos = x_counts.get(1, 0)
    x_neg = x_counts.get(-1, 0)
    x_abs = x_counts.get(0, 0)
    logging.info(
        f"Service quality (X) distribution — "
        f"Positive: {x_pos}, Negative: {x_neg}, Absent: {x_abs}"
    )

    # ---- Save lean output (only columns needed for OLS regression) ----
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out = pd.DataFrame({
        'review_rating':        df['review_rating'],
        'satisfaction_raw_M':   df['satisfaction_raw_M'],
        'customer_satisfaction_M': df['customer_satisfaction_M'],
        'service_quality_X':    df['service_quality_X'],
    })
    df_out.to_csv(output_file, index=False)
    logging.info(f"Output saved to: {output_file}")

    log_step(
        'Step 7 - Sentiment Scoring',
        f"{total_rows:,} rows scored | "
        f"M (customer satisfaction): {pos_count} pos / {neg_count} neg / {neu_count} neu | "
        f"X (ABSA): {x_pos} pos / {x_neg} neg / {x_abs} absent"
    )


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'data', 'output'), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(
                os.path.join(BASE_DIR, 'data', 'output', 'step7_sentiment_scoring.log'),
                encoding='utf-8'
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )

    run_sentiment_scoring(INPUT_FILE, OUTPUT_FILE, POS_LEXICON_FILE, NEG_LEXICON_FILE, SERVICE_KEYWORDS_FILE)
