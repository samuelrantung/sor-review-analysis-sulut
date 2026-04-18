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
# Load API Key from environment variable for better security
# In your terminal, run: export OUTSCRAPER_API_KEY='your_key_here'
load_dotenv()
API_KEY = os.getenv('OUTSCRAPER_API_KEY')

# Research Parameters
# Using a set to automatically handle duplicates and adding 'Bitung'
# Using Indonesian for queries to match the 'language' parameter
QUERIES = list(set([
    'Restoran di Manado, Sulawesi Utara',
    'Restoran di Tomohon, Sulawesi Utara',
    'Restoran di Bitung, Sulawesi Utara'
]))

# Limits for testing vs. production.
# For production, you might set PLACES_LIMIT_PER_QUERY to 50 and REVIEWS_LIMIT_PER_PLACE to 100.
PLACES_LIMIT_PER_QUERY = 45 # Oversample: e.g., 1.5 * 30 target places per city
REVIEWS_LIMIT_PER_PLACE = 30 # Get up to 50 reviews per place
TARGET_TOTAL_REVIEWS = 1600 # Stop extraction once we have this many valid reviews total

# Specific Place ID for testing (set to None to run normal search)
# Example: "ChIJxwJDjQ5thzIRinWjKfRRCXs"
TEST_PLACE_ID = None

# Execution Mode: 'search_only', 'reviews_only', or 'all'
RUN_MODE = 'reviews_only'

# Required if RUN_MODE = 'reviews_only'
# Path to the CSV file containing place_id and name columns from a previous search run
PLACES_INPUT_FILE = "data/raw/20260304_105910/places_search_filtered.csv"

# Date range for the research (UTC)
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2026, 2, 1, tzinfo=timezone.utc)  # Up to, but not including, Feb 1, 2026

# Output file configuration
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join("data/raw", timestamp)
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "all_reviews_raw.csv")

# ==========================================
# 2. API CLIENT SETUP
# ==========================================
def get_api_client():
    """Initializes and returns the Outscraper API client, exiting if the key is missing."""
    if not API_KEY:
        logging.error("OUTSCRAPER_API_KEY environment variable not set.")
        logging.error("Please set it before running the script (e.g., export OUTSCRAPER_API_KEY='your_key').")
        sys.exit(1)  # Exit with an error code
    return ApiClient(API_KEY)


# ==========================================
# 3. THE EXTRACTION ENGINE
# ==========================================
def run_outscraper_extraction(client, queries, places_limit, reviews_limit, output_path, run_mode='all', places_input_file=None, target_total_reviews=None):
    """
    Extracts Google Maps reviews in a robust, two-stage process.
    1. Searches for places based on queries.
    2. Fetches reviews for each place individually and saves incrementally.
    """
    # Ensure the output directory exists before any file operations
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if target_total_reviews is None:
        target_total_reviews = 1600

    # --- STAGE 1: Find all candidate places ---
    unique_places = []

    if run_mode in ['search_only', 'all']:
        if TEST_PLACE_ID:
            logging.info(f"Stage 1: Skipped (Test Mode). Using specific Place ID: {TEST_PLACE_ID}")
            unique_places = [{'place_id': TEST_PLACE_ID, 'name': 'Manual Test Place'}]
        else:
            logging.info(f"Stage 1: Searching for up to {places_limit} places per query...")
            
            try:
                # Batch request: Sending all queries at once.
                # Outscraper processes these in parallel (internal async).
                search_results = client.google_maps_search(
                    queries,
                    limit=places_limit,
                    drop_duplicates=True, # API-side deduplication
                    region='ID',         # Focus on Indonesia
                    language='id'
                )
                # Flatten the list of lists (one list per query) into a single list of places
                if search_results and isinstance(search_results[0], list):
                    unique_places = [place for sublist in search_results for place in sublist]
                elif search_results:
                    unique_places = search_results
                logging.info(f"Found {len(unique_places)} places (deduplicated by API).")

                # Save search results immediately to prevent data loss
                if unique_places:
                    search_save_path = os.path.join(
                        os.path.dirname(output_path), 
                        "places_search_raw.csv"
                    )
                    pd.DataFrame(unique_places).to_csv(search_save_path, index=False, encoding='utf-8-sig')
                    logging.info(f"Search results saved immediately to: {os.path.abspath(search_save_path)}")
                    
                    # Save raw JSON for debugging
                    search_json_path = os.path.join(
                        os.path.dirname(output_path), 
                        "places_search_raw.json"
                    )
                    with open(search_json_path, 'w', encoding='utf-8') as f:
                        json.dump(unique_places, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.error(f"Error during batch search: {e}")
                unique_places = []
        
        if run_mode == 'search_only':
            logging.info("Search complete. Exiting as RUN_MODE is 'search_only'.")
            return

    # --- STAGE 2 PREP: Load places if running reviews_only ---
    if run_mode == 'reviews_only':
        if TEST_PLACE_ID:
             unique_places = [{'place_id': TEST_PLACE_ID, 'name': 'Manual Test Place'}]
        elif places_input_file and os.path.exists(places_input_file):
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
        logging.warning("No places found from initial search. Exiting.")
        return

    # --- STAGE 1.5: Group, Sort, and Interleave places for balanced city representation ---
    logging.info("Grouping places by city for balanced extraction...")

    # Dynamically create keys from queries to be more robust
    # e.g., 'Restoran di Manado, ...' -> 'manado'
    city_keys = [q.split(',')[0].split(' di ')[1].lower().strip() for q in QUERIES]
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

    # Sort each city's list by review count
    for city, places_list in places_by_city.items():
        places_list.sort(key=lambda p: int(p.get('reviews', 0) or 0), reverse=True)
        logging.info(f"Found and sorted {len(places_list)} places for '{city}'.")
    if other_places:
        logging.warning(f"Found {len(other_places)} places that did not match a target city.")

    # Interleave the lists to create a balanced processing order (round-robin)
    interleaved_places = []
    max_len = 0
    if city_keys:
        max_len = max(len(places_by_city.get(city, [])) for city in city_keys)

    for i in range(max_len):
        for city_key in city_keys:
            if i < len(places_by_city.get(city_key, [])):
                interleaved_places.append(places_by_city[city_key][i])
    
    logging.info(f"Created a balanced, interleaved list of {len(interleaved_places)} places to process.")

    # --- STAGE 2: Fetch reviews for each place individually ---
    logging.info(f"Stage 2: Fetching up to {reviews_limit} reviews per place...")
    logging.info(f"Filtering reviews newer than {START_DATE.date()} from the API.")
    total_reviews_saved = 0
    file_header_written = False

    # Convert start date to a Unix timestamp for the API's 'cutoff' parameter
    start_timestamp = int(START_DATE.timestamp())
    end_timestamp = int(END_DATE.timestamp())

    # Loop through each unique place found in Stage 1
    pbar = tqdm(interleaved_places, desc="Fetching Reviews", unit="place")
    for place_data in pbar:
        # Check if we hit the global target
        if total_reviews_saved >= target_total_reviews:
            logging.info(f"🎉 TARGET REACHED: Collected {total_reviews_saved} reviews (Target: {target_total_reviews}). Stopping.")
            break

        place_id = place_data.get('place_id')
        place_name = place_data.get('name', 'N/A')
        if not place_id:
            continue

        pbar.set_description(f"Processing: {place_name[:30]}")
        try:
            # Make a small, targeted API call for reviews of this single place
            logging.debug(f"Requesting reviews for {place_name} ({place_id}). Limit: {reviews_limit}, Cutoff: {start_timestamp}, Start: {end_timestamp}")
            location_reviews = client.google_maps_reviews(
                place_id, # Use the specific place_id
                reviews_limit=reviews_limit,
                language='id',
                sort='newest',  # Useful for getting current data
                cutoff=start_timestamp, # API-side filtering for start date
                start=end_timestamp,    # API-side filtering for end date (newest)
                ignore_empty=True, # Skip places with no reviews
                region='ID'       # Focus on Indonesia for reviews
            )

            # Save raw JSON immediately for backup (even if empty)
            raw_reviews_dir = os.path.join(os.path.dirname(output_path), "reviews_raw_json")
            os.makedirs(raw_reviews_dir, exist_ok=True)
            # Create a safe filename
            safe_name = "".join([c if c.isalnum() else "_" for c in place_name])[:50]
            json_filename = f"{safe_name}_{place_id}.json"

            with open(os.path.join(raw_reviews_dir, json_filename), "w", encoding="utf-8") as f:
                json.dump(location_reviews, f, ensure_ascii=False, indent=2)
            logging.info(f"Raw JSON saved to: {os.path.basename(json_filename)}")

            # Handle potential list response from API (Outscraper usually returns a list)
            # Flatten list of lists if necessary (e.g. [[{...}]])
            if isinstance(location_reviews, list) and location_reviews and isinstance(location_reviews[0], list):
                location_reviews = location_reviews[0]

            # Get the first place object
            data_to_process = location_reviews[0] if isinstance(location_reviews, list) and location_reviews else location_reviews
            if isinstance(data_to_process, list): data_to_process = None # Safety check if structure is unexpected

            if not data_to_process:
                logging.info(f"Skipped '{place_name}' (no reviews found or API returned empty).")
                continue

            # The result is a single dictionary for the location
            address = data_to_process.get('full_address', 'N/A')
            # Safety: Use 'or []' because get() returns None if the JSON value is null
            reviews_list = data_to_process.get('reviews_data') or []
            reviews_for_location_in_range = []
            
            for rev in reviews_list:
                review_date_str = rev.get('review_datetime_utc')
                if not review_date_str:
                    logging.debug(f"Skipping review {rev.get('review_id', 'unknown')}: No date found.")
                    continue  # Skip reviews without a date

                try:
                    # Parse the date; pd.to_datetime handles ISO format with 'Z' correctly
                    review_date = pd.to_datetime(review_date_str, utc=True).to_pydatetime()
                except (ValueError, TypeError):
                    logging.debug(f"Skipping review {rev.get('review_id', 'unknown')}: Malformed date '{review_date_str}'.")
                    continue # Skip if date is malformed

                # The API's 'cutoff' parameter handles the start date.
                # We only need to check for the end date here as a safeguard.
                if not (review_date < END_DATE):
                    logging.debug(f"Skipping review {rev.get('review_id', 'unknown')}: Date {review_date} is past end date {END_DATE}.")
                    continue # Skip reviews that are past our research end date

                # Get raw text
                raw_text = rev.get('review_text')

                # This review is valid, create the row
                row = {
                    'place_name': place_name,
                    'address': address,
                    'review_author': rev.get('author_title'),
                    'review_text': raw_text,
                    'review_rating': rev.get('review_rating'),
                    'review_date': review_date_str,
                    'google_id': place_id
                }
                reviews_for_location_in_range.append(row)

            # Incrementally save the filtered reviews for this location to prevent data loss
            if reviews_for_location_in_range:
                df_chunk = pd.DataFrame(reviews_for_location_in_range)
                
                # Append to CSV. Write header only for the very first chunk.
                df_chunk.to_csv(
                    output_path,
                    mode='a',
                    header=not file_header_written,
                    index=False,
                    encoding='utf-8-sig'
                )
                
                if not file_header_written:
                    file_header_written = True

                num_saved = len(df_chunk)
                total_reviews_saved += num_saved
                logging.info(f"Saved {num_saved} reviews for '{place_name}'. Total saved: {total_reviews_saved}")

        except Exception as e:
            logging.error(f"An error occurred while fetching reviews for '{place_name}': {e}")
            continue # Move to the next place

    # Final summary
    logging.info("="*20)
    if total_reviews_saved > 0:
        logging.info(f"Extraction Complete!")
        logging.info(f"Total Rows Captured: {total_reviews_saved}")
        logging.info(f"All data saved to: {os.path.abspath(output_path)}")
    else:
        logging.warning("No reviews were found within the specified date range and other criteria.")
    logging.info("="*20)


if __name__ == "__main__":
    # Ensure output directory exists for logs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Configure logging
    log_filename = os.path.join(OUTPUT_DIR, "extraction_debug.log")
    logging.basicConfig(
        level=logging.DEBUG, # Set to DEBUG to capture all details
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'), # Save logs to file
            logging.StreamHandler(sys.stdout)  # Print to console
        ]
    )
    # Enable detailed request logging from urllib3 (used by requests/outscraper)
    logging.getLogger("urllib3").setLevel(logging.DEBUG)

    api_client = get_api_client()
    run_outscraper_extraction(
        client=api_client,
        queries=QUERIES,
        places_limit=PLACES_LIMIT_PER_QUERY,
        reviews_limit=REVIEWS_LIMIT_PER_PLACE,
        output_path=OUTPUT_FILENAME,
        run_mode=RUN_MODE,
        places_input_file=PLACES_INPUT_FILE,
        target_total_reviews=TARGET_TOTAL_REVIEWS
    )