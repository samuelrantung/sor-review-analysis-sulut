import pandas as pd
from outscraper import ApiClient
import os
import sys
from datetime import datetime, timezone
import json
import logging
from tqdm import tqdm
from dotenv import load_dotenv


# ==========================================
# 1. CONFIGURATION
# ==========================================
load_dotenv()
API_KEY = os.getenv('OUTSCRAPER_API_KEY')

# Base directory (pipeline/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Research Parameters
QUERIES = [
    'Restoran di Manado, Sulawesi Utara',
    'Restoran di Tomohon, Sulawesi Utara',
    'Restoran di Bitung, Sulawesi Utara'
]

PLACES_LIMIT_PER_QUERY = 45   # Oversample: 1.5x of 30 target places per city
REVIEWS_LIMIT_PER_PLACE = 30  # Max reviews per place
TARGET_TOTAL_REVIEWS = 1600   # Stop extraction once this total is reached

# Execution Mode:
# 'all'          : Full run — search for places then fetch reviews (default)
# 'search_only'  : Only search for places, save results to CSV
# 'reviews_only' : Skip search, load places from PLACES_INPUT_FILE and fetch reviews
RUN_MODE = 'all'

# Required if RUN_MODE = 'reviews_only'
PLACES_INPUT_FILE = os.path.join(BASE_DIR, 'data', 'input', 'places_search_filtered.csv')

# Date range for the research (UTC)
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 2, 1, tzinfo=timezone.utc)

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'output')
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, 'all_reviews_raw.csv')


# ==========================================
# 2. API CLIENT SETUP
# ==========================================
def get_api_client():
    """Initializes and returns the Outscraper API client, exiting if the key is missing."""
    if not API_KEY:
        logging.error("OUTSCRAPER_API_KEY environment variable not set.")
        logging.error("Please set it before running the script (e.g., export OUTSCRAPER_API_KEY='your_key').")
        sys.exit(1)
    return ApiClient(API_KEY)


# ==========================================
# 3. THE EXTRACTION ENGINE
# ==========================================
def run_outscraper_extraction(
    client,
    queries,
    places_limit,
    reviews_limit,
    output_path,
    start_date,
    end_date,
    run_mode='all',
    places_input_file=None,
    target_total_reviews=TARGET_TOTAL_REVIEWS
):
    """
    Extracts Google Maps reviews in a two-stage process.
    Stage 1: Search for candidate places based on queries.
    Stage 2: Fetch reviews for each place individually and save incrementally.
    """
    valid_modes = ['all', 'search_only', 'reviews_only']
    if run_mode not in valid_modes:
        logging.error(f"Invalid RUN_MODE '{run_mode}'. Must be one of: {valid_modes}")
        sys.exit(1)

    logging.info(f"Date range: {start_date.date()} to {end_date.date()} (UTC)")
    logging.info(f"Run mode  : {run_mode}")
    logging.info(f"Target    : {target_total_reviews} reviews")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # --- STAGE 1: Find all candidate places ---
    unique_places = []

    if run_mode in ['search_only', 'all']:
        logging.info(f"Stage 1: Searching for up to {places_limit} places per query...")

        try:
            search_results = client.google_maps_search(
                queries,
                limit=places_limit,
                drop_duplicates=True,
                region='ID',
                language='id'
            )
            # Flatten list of lists (one list per query) into a single list
            if search_results and isinstance(search_results[0], list):
                unique_places = [place for sublist in search_results for place in sublist]
            elif search_results:
                unique_places = search_results
            logging.info(f"Found {len(unique_places)} places.")

            # Save search results immediately to prevent data loss
            if unique_places:
                search_save_path = os.path.join(os.path.dirname(output_path), 'places_search_raw.csv')
                pd.DataFrame(unique_places).to_csv(search_save_path, index=False, encoding='utf-8-sig')
                logging.info(f"Search results saved to: {os.path.abspath(search_save_path)}")

                search_json_path = os.path.join(os.path.dirname(output_path), 'places_search_raw.json')
                with open(search_json_path, 'w', encoding='utf-8') as f:
                    json.dump(unique_places, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error during place search: {e}")
            unique_places = []

        if run_mode == 'search_only':
            logging.info("Search complete. Exiting as RUN_MODE is 'search_only'.")
            return

    # --- STAGE 2 PREP: Load places if running reviews_only ---
    if run_mode == 'reviews_only':
        if places_input_file and os.path.exists(places_input_file):
            logging.info(f"Loading places from: {places_input_file}")
            try:
                df_input = pd.read_csv(places_input_file)
                unique_places = df_input.to_dict('records')
            except Exception as e:
                logging.error(f"Failed to read input file: {e}")
                return
        else:
            logging.error(f"RUN_MODE='reviews_only' but PLACES_INPUT_FILE is invalid: {places_input_file}")
            return

    if not unique_places:
        logging.warning("No places found. Exiting.")
        return

    # --- STAGE 1.5: Group, sort, and interleave places for balanced city representation ---
    logging.info("Grouping places by city for balanced extraction...")

    # Extract city keys from queries, e.g. 'Restoran di Manado, ...' -> 'manado'
    city_keys = [q.split(',')[0].split(' di ')[1].lower().strip() for q in queries]
    places_by_city = {key: [] for key in city_keys}
    other_places = []

    for place in unique_places:
        address_string = f"{place.get('city', '')} {place.get('full_address', '')}".lower()
        matched = False
        for city_key in city_keys:
            if city_key in address_string:
                places_by_city[city_key].append(place)
                matched = True
                break
        if not matched:
            other_places.append(place)

    # Sort each city's places by review count (descending) to prioritize popular restaurants
    for city, places_list in places_by_city.items():
        places_list.sort(key=lambda p: int(p.get('reviews', 0) or 0), reverse=True)
        logging.info(f"Found and sorted {len(places_list)} places for '{city}'.")
    if other_places:
        logging.warning(f"Found {len(other_places)} places that did not match a target city.")

    # Interleave city lists using round-robin to ensure balanced representation across cities
    interleaved_places = []
    if city_keys:
        max_len = max(len(places_by_city.get(city, [])) for city in city_keys)
        for i in range(max_len):
            for city_key in city_keys:
                if i < len(places_by_city.get(city_key, [])):
                    interleaved_places.append(places_by_city[city_key][i])

    logging.info(f"Processing {len(interleaved_places)} places (balanced across cities).")

    # --- STAGE 2: Fetch reviews for each place ---
    logging.info(f"Stage 2: Fetching up to {reviews_limit} reviews per place...")

    # Resume support: load already-processed place_ids from existing output CSV
    processed_place_ids = set()
    file_header_written = False
    if os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            if 'google_id' in df_existing.columns:
                processed_place_ids = set(df_existing['google_id'].dropna().unique())
                total_reviews_saved = len(df_existing)
                file_header_written = True
                logging.info(f"Resuming: found {len(processed_place_ids)} already-processed places ({total_reviews_saved} reviews). Skipping them.")
        except Exception as e:
            logging.warning(f"Could not read existing output file for resume: {e}. Starting fresh.")
            total_reviews_saved = 0
    else:
        total_reviews_saved = 0

    # Convert date range to Unix timestamps for the Outscraper API
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    pbar = tqdm(interleaved_places, desc="Fetching Reviews", unit="place")
    for place_data in pbar:
        if total_reviews_saved >= target_total_reviews:
            logging.info(f"Target reached: {total_reviews_saved} reviews collected. Stopping.")
            break

        place_id = place_data.get('place_id')
        place_name = place_data.get('name', 'N/A')
        if not place_id:
            continue

        # Skip places already processed in a previous run
        if place_id in processed_place_ids:
            pbar.set_description(f"Skipping (done): {place_name[:25]}")
            continue

        pbar.set_description(f"Processing: {place_name[:30]}")
        try:
            location_reviews = client.google_maps_reviews(
                place_id,
                reviews_limit=reviews_limit,
                language='id',
                sort='newest',
                cutoff=start_timestamp,  # API-side filtering: exclude reviews before START_DATE
                start=end_timestamp,     # API-side filtering: exclude reviews after END_DATE
                ignore_empty=True,
                region='ID'
            )

            # Save raw JSON per place as backup
            raw_reviews_dir = os.path.join(os.path.dirname(output_path), 'reviews_raw_json')
            os.makedirs(raw_reviews_dir, exist_ok=True)
            safe_name = "".join([c if c.isalnum() else "_" for c in place_name])[:50]
            json_filename = f"{safe_name}_{place_id}.json"
            with open(os.path.join(raw_reviews_dir, json_filename), "w", encoding="utf-8") as f:
                json.dump(location_reviews, f, ensure_ascii=False, indent=2)

            # Normalize API response: flatten list of lists if necessary
            if isinstance(location_reviews, list) and location_reviews and isinstance(location_reviews[0], list):
                location_reviews = location_reviews[0]

            data_to_process = location_reviews[0] if isinstance(location_reviews, list) and location_reviews else location_reviews
            if isinstance(data_to_process, list):
                data_to_process = None  # Unexpected structure

            if not data_to_process:
                logging.info(f"Skipped '{place_name}' (no reviews returned).")
                continue

            address = data_to_process.get('full_address', 'N/A')
            # Use 'or []' to handle cases where reviews_data is explicitly null in the API response
            reviews_list = data_to_process.get('reviews_data') or []
            reviews_for_location_in_range = []

            for rev in reviews_list:
                review_date_str = rev.get('review_datetime_utc')
                if not review_date_str:
                    continue

                try:
                    review_date = pd.to_datetime(review_date_str, utc=True).to_pydatetime()
                except (ValueError, TypeError):
                    continue

                # Safeguard: ensure review falls within end_date (API cutoff handles start_date)
                if review_date >= end_date:
                    continue

                row = {
                    'place_name': place_name,
                    'address': address,
                    'review_author': rev.get('author_title'),
                    'review_text': rev.get('review_text'),
                    'review_rating': rev.get('review_rating'),
                    'review_date': review_date_str,
                    'google_id': place_id
                }
                reviews_for_location_in_range.append(row)

            # Append to CSV incrementally to prevent data loss
            if reviews_for_location_in_range:
                df_chunk = pd.DataFrame(reviews_for_location_in_range)
                df_chunk.to_csv(
                    output_path,
                    mode='a',
                    header=not file_header_written,
                    index=False,
                    encoding='utf-8-sig'
                )
                file_header_written = True
                total_reviews_saved += len(df_chunk)
                logging.info(f"Saved {len(df_chunk)} reviews for '{place_name}'. Total: {total_reviews_saved}")

        except Exception as e:
            logging.error(f"Error fetching reviews for '{place_name}': {e}")
            continue

    # Final summary
    logging.info("=" * 40)
    if total_reviews_saved > 0:
        logging.info(f"Extraction complete. Total reviews saved: {total_reviews_saved}")
        logging.info(f"Output: {os.path.abspath(output_path)}")
    else:
        logging.warning("No reviews were collected within the specified date range.")
    logging.info("=" * 40)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, "step1_scraping.log"), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    api_client = get_api_client()
    run_outscraper_extraction(
        client=api_client,
        queries=QUERIES,
        places_limit=PLACES_LIMIT_PER_QUERY,
        reviews_limit=REVIEWS_LIMIT_PER_PLACE,
        output_path=OUTPUT_FILENAME,
        start_date=START_DATE,
        end_date=END_DATE,
        run_mode=RUN_MODE,
        places_input_file=PLACES_INPUT_FILE,
        target_total_reviews=TARGET_TOTAL_REVIEWS
    )
