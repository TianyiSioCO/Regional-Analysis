"""
Batch download New York State NSRDB solar radiation data
Supports two modes:
  1. TMY mode: Download Typical Meteorological Year data (years may vary by location)
  2. PSM single-year mode: Download actual data for a specified year (consistent timeline across all points)

Uses polygon boundaries to define region, generates grid coordinates, and downloads data point by point
"""

import requests
import zipfile
import time
from pathlib import Path
from datetime import datetime
import json
import threading
from queue import Queue

# ============ Configuration Parameters ============
# Data mode selection
# - "tmy": Download TMY Typical Meteorological Year data (default, but years may vary by location)
# - "single_year": Download PSM data for specified single year (recommended, consistent timeline)
DATA_MODE = "single_year"  # Options: "tmy" or "single_year"

# Single year mode settings (effective when DATA_MODE="single_year")
# Available year range: 1998-2023 (Continental US)
SINGLE_YEAR = 2020  # Recommended to select a meteorologically normal year

# API endpoint configuration
# Note: Using .csv direct download mode to avoid 404 issues with async ZIP mode
API_ENDPOINTS = {
    "tmy": "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-tmy-v4-0-0-download.csv",
    "single_year": "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv",
}

# Multiple API Key configuration - Supports credential rotation
# Note: NREL API has stricter email validation, requires meaningful email format
API_CREDENTIALS = [
    {"api_key": "aubcOxuI74skYBd80St05JVm0zdhtjDzV0RLObUL", "email": "solar.data.user1@outlook.com"},
    {"api_key": "0EuGhvS3LTYgI90EkLbtgg0IkIw20jxgyWrGn7qY", "email": "nsrdb.download2@outlook.com"},
    {
        "api_key": "2mF0rdaGSTpixNCiZAFrk1CCnIPBFJIljQSfkLKy",
        "email": "gis.research3@outlook.com",
    },
    {
        "api_key": "seyN9FCqZno8aekJchj9qSM7iVkyNqOH17BGNRmG",
        "email": "energy.analysis4@outlook.com",
    },
    {
        "api_key": "XWkg8fPGzrxRZ6vRUnT0xQBijle7mfQhFvE24cI4",
        "email": "solar.project5@outlook.com",
    },
    {"api_key": "tk0uzHlcxiNk93WSLsBkeLuvVKFuuAjDB2QG7qSm", "email": "pv.modeling6@outlook.com"},
    {"api_key": "Lvp7FQbtoPnOoam28RbdXeHBxz3lZ0L5Q7BUdO7O", "email": "renewable.data7@outlook.com"},
    {
        "api_key": "2mF0rdaGSTpixNCiZAFrk1CCnIPBFJIljQSfkLKy",
        "email": "climate.study8@outlook.com",
    },
    {
        "api_key": "JIN3fdlv1cmYQxLw7sQKmfA3Pwf7NCqnwafqx353",
        "email": "irradiance.map9@outlook.com",
    },
    {
        "api_key": "fTq1aPuILE52gKQbfhFL2rNgbKmMe86QkVACv1qE",
        "email": "weather.analysis10@outlook.com",
    },
]

# API Key cooldown configuration
COOLDOWN_DURATION = 60  # Individual key cooldown duration (seconds)
ALL_KEYS_COOLDOWN_WAIT = 30  # Wait time when all keys are in cooldown (seconds)

# Alaska simplified boundary polygon vertices (longitude, latitude) - Continental main body
# Note: Alaska is huge, recommend increasing grid resolution or limiting download points
AK_POLYGON = [
    # Southeast coast (starting from Canadian border)
    (-130.0, 54.7),
    (-130.5, 55.3),
    (-131.0, 55.8),
    (-132.0, 56.5),
    (-133.0, 57.0),
    (-134.0, 58.0),
    (-135.0, 58.5),
    (-136.0, 59.0),
    (-137.5, 59.0),
    (-139.0, 59.5),
    (-140.5, 60.0),
    # Eastern boundary (Canadian border -141° meridian)
    (-141.0, 60.0),
    (-141.0, 62.0),
    (-141.0, 64.0),
    (-141.0, 66.0),
    (-141.0, 68.0),
    (-141.0, 69.7),
    # Northern coast (Arctic Ocean)
    (-143.0, 70.0),
    (-146.0, 70.1),
    (-148.0, 70.3),
    (-150.0, 70.5),
    (-152.0, 70.8),
    (-154.0, 71.0),
    (-156.5, 71.3),  # Near Point Barrow, northernmost point
    (-158.0, 70.8),
    (-160.0, 70.5),
    (-162.0, 70.0),
    (-164.0, 69.5),
    (-166.0, 68.9),
    # Western coast (Bering Sea)
    (-167.0, 68.0),
    (-166.5, 66.5),
    (-165.0, 65.0),
    (-164.0, 64.0),
    (-163.0, 63.0),
    (-162.0, 62.0),
    (-163.0, 61.0),
    (-164.0, 60.5),
    (-165.0, 60.5),
    (-166.0, 60.0),
    (-165.5, 59.0),
    (-164.0, 58.5),
    (-162.0, 58.5),
    (-160.0, 58.8),
    (-158.0, 58.5),
    (-157.0, 58.0),
    (-156.0, 57.0),
    (-155.0, 56.5),
    # Southern coast (Gulf of Alaska)
    (-154.0, 56.5),
    (-153.0, 57.0),
    (-152.0, 57.5),
    (-151.0, 58.5),
    (-150.0, 59.0),
    (-149.0, 59.5),
    (-148.0, 60.0),
    (-147.0, 60.5),
    (-146.0, 60.3),
    (-145.0, 60.2),
    (-144.0, 60.0),
    (-143.0, 59.5),
    (-142.0, 59.5),
    (-141.0, 59.5),
    (-140.0, 59.8),
    (-139.0, 59.5),
    (-138.0, 59.0),
    (-137.0, 58.5),
    (-136.5, 58.0),
    (-135.5, 57.0),
    (-134.5, 56.5),
    (-133.5, 56.0),
    (-132.5, 55.5),
    (-131.5, 55.0),
    (-130.0, 54.7),  # Close polygon
]

# New York State precise boundary polygon vertices (longitude, latitude)
NY_POLYGON = [
    (-79.7624, 42.5142),
    (-79.0672, 42.7783),
    (-78.9313, 42.8508),
    (-78.9024, 42.9061),
    (-78.9313, 42.9554),
    (-78.9656, 42.9584),
    (-79.0219, 42.9886),
    (-79.0027, 43.0568),
    (-79.0727, 43.0769),
    (-79.0713, 43.1220),
    (-79.0302, 43.1441),
    (-79.0576, 43.1801),
    (-79.0604, 43.2482),
    (-79.0837, 43.2812),
    (-79.2004, 43.4509),
    (-78.6909, 43.6311),
    (-76.7958, 43.6321),
    (-76.4978, 43.9987),
    (-76.4388, 44.0965),
    (-76.3536, 44.1349),
    (-76.3124, 44.1989),
    (-76.2437, 44.2049),
    (-76.1655, 44.2413),
    (-76.1353, 44.2973),
    (-76.0474, 44.3327),
    (-75.9856, 44.3553),
    (-75.9196, 44.3749),
    (-75.8730, 44.3994),
    (-75.8221, 44.4308),
    (-75.8098, 44.4740),
    (-75.7288, 44.5425),
    (-75.5585, 44.6647),
    (-75.4088, 44.7672),
    (-75.3442, 44.8101),
    (-75.3058, 44.8383),
    (-75.2399, 44.8676),
    (-75.1204, 44.9211),
    (-74.9995, 44.9609),
    (-74.9899, 44.9803),
    (-74.9103, 44.9852),
    (-74.8856, 45.0017),
    (-74.8306, 45.0153),
    (-74.7633, 45.0046),
    (-74.7070, 45.0027),
    (-74.5642, 45.0007),
    (-74.1467, 44.9920),
    (-73.7306, 45.0037),
    (-73.4203, 45.0085),
    (-73.3430, 45.0109),
    (-73.3547, 44.9874),
    (-73.3379, 44.9648),
    (-73.3396, 44.9160),
    (-73.3739, 44.8354),
    (-73.3324, 44.8013),
    (-73.3667, 44.7419),
    (-73.3873, 44.6139),
    (-73.3736, 44.5787),
    (-73.3049, 44.4916),
    (-73.2953, 44.4289),
    (-73.3365, 44.3513),
    (-73.3118, 44.2757),
    (-73.3818, 44.1980),
    (-73.4079, 44.1142),
    (-73.4367, 44.0511),
    (-73.4065, 44.0165),
    (-73.4079, 43.9375),
    (-73.3749, 43.8771),
    (-73.3914, 43.8167),
    (-73.3557, 43.7790),
    (-73.4244, 43.6460),
    (-73.4340, 43.5893),
    (-73.3969, 43.5655),
    (-73.3818, 43.6112),
    (-73.3049, 43.6271),
    (-73.3063, 43.5764),
    (-73.2582, 43.5675),
    (-73.2445, 43.5227),
    (-73.2582, 43.2582),
    (-73.2733, 42.9715),
    (-73.2898, 42.8004),
    (-73.2664, 42.7460),
    (-73.3708, 42.4630),
    (-73.5095, 42.0840),
    (-73.4903, 42.0218),
    (-73.4999, 41.8808),
    (-73.5535, 41.2953),
    (-73.4834, 41.2128),
    (-73.7275, 41.1011),
    (-73.6644, 41.0237),
    (-73.6578, 40.9851),
    (-73.6132, 40.9509),
    (-72.4823, 41.1869),
    (-72.0950, 41.2551),
    (-71.9714, 41.3005),
    (-71.9193, 41.3108),
    (-71.7915, 41.1838),
    (-71.7929, 41.1249),
    (-71.7517, 41.0462),
    (-72.9465, 40.6306),
    (-73.4628, 40.5368),
    (-73.8885, 40.4887),
    (-73.9490, 40.5232),
    (-74.2271, 40.4772),
    (-74.2532, 40.4861),
    (-74.1866, 40.6468),
    (-74.0547, 40.6556),
    (-74.0156, 40.7618),
    (-73.9421, 40.8699),
    (-73.8934, 40.9980),
    (-73.9854, 41.0343),
    (-74.6274, 41.3268),
    (-74.7084, 41.3583),
    (-74.7101, 41.3811),
    (-74.8265, 41.4386),
    (-74.9913, 41.5075),
    (-75.0668, 41.6000),
    (-75.0366, 41.6719),
    (-75.0545, 41.7672),
    (-75.1945, 41.8808),
    (-75.3552, 42.0013),
    (-75.4266, 42.0003),
    (-77.0306, 42.0013),
    (-79.7250, 41.9993),
    (-79.7621, 42.0003),
    (-79.7621, 42.1827),
    (-79.7621, 42.5146),
    (-79.7624, 42.5142),
]

# Grid resolution (degrees) - 0.095° ≈ 9.2km, approximately 1700 points
GRID_RESOLUTION = 0.09

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "AnalyzeData" / "data"

# Download limits
MAX_POINTS = None  # Set to None for no limit, generates all points within polygon bounds
REQUEST_INTERVAL = 2.1  # Request interval (seconds), API requires 1 per 2 seconds
DOWNLOAD_INITIAL_WAIT = 10  # Initial download wait time (seconds) - Can be shortened with multiple keys
DOWNLOAD_RETRY_WAIT = 3  # Retry wait time (seconds) - Fast polling check
MAX_DOWNLOAD_RETRIES = 100  # Maximum retry attempts - Shorter wait time requires more retries
DOWNLOAD_THREADS = 10  # Parallel download threads - Multiple keys support higher concurrency
# ==================================


class ApiKeyManager:
    """
    API Key rotation manager
    - Supports rotation of multiple API Keys
    - Automatically switches and marks cooldown on 429/400 errors
    - Waits and retries when all keys are in cooldown
    """

    def __init__(self, credentials):
        self.credentials = credentials
        self.current_index = 0
        self.cooldown_until = {}  # {index: cooldown_end_timestamp}
        self.lock = threading.Lock()

    def get_current(self):
        """Get currently available API Key and email"""
        with self.lock:
            return self._get_available_credential()

    def _get_available_credential(self):
        """Internal method: Get available credential, wait if all are in cooldown"""
        while True:
            now = time.time()
            available_indices = []

            # Find all available keys
            for i in range(len(self.credentials)):
                cooldown_end = self.cooldown_until.get(i, 0)
                if now >= cooldown_end:
                    available_indices.append(i)

            if available_indices:
                # Prefer current index, use first available if current is unavailable
                if self.current_index in available_indices:
                    idx = self.current_index
                else:
                    idx = available_indices[0]
                    self.current_index = idx

                cred = self.credentials[idx]
                return {
                    "index": idx,
                    "api_key": cred["api_key"],
                    "email": cred["email"],
                }
            else:
                # All keys in cooldown, wait and retry
                min_wait = min(self.cooldown_until.values()) - now
                wait_time = max(min_wait, ALL_KEYS_COOLDOWN_WAIT)
                print(f"\n⚠️  All API Keys in cooldown, waiting {wait_time:.0f} seconds before retry...")
                self.lock.release()
                time.sleep(wait_time)
                self.lock.acquire()

    def mark_cooldown(self, index):
        """Mark specified key as in cooldown state"""
        with self.lock:
            self.cooldown_until[index] = time.time() + COOLDOWN_DURATION
            print(f"  ⏸️  API Key #{index + 1} entered cooldown period ({COOLDOWN_DURATION}s)")

    def switch_to_next(self, silent=True):
        """
        Switch to next available key
        silent: Silent mode, no print for switching (used during normal rotation)
        """
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.credentials)
            if not silent:
                print(f"  🔄 Switched to API Key #{self.current_index + 1}")

    def get_status(self):
        """Get status of all keys"""
        with self.lock:
            now = time.time()
            status = []
            for i, cred in enumerate(self.credentials):
                cooldown_end = self.cooldown_until.get(i, 0)
                if now >= cooldown_end:
                    status.append(f"Key#{i+1}: Available")
                else:
                    remaining = int(cooldown_end - now)
                    status.append(f"Key#{i+1}: Cooldown({remaining}s)")
            return ", ".join(status)


# Global API Key manager instance
api_key_manager = ApiKeyManager(API_CREDENTIALS)


def point_in_polygon(lon, lat, polygon):
    """
    Ray casting algorithm to determine if point is inside polygon
    polygon: List of [(lon1, lat1), (lon2, lat2), ...]
    """
    n = len(polygon)
    inside = False

    x, y = lon, lat
    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def get_polygon_bounds(polygon):
    """Get bounding box of polygon"""
    lons = [p[0] for p in polygon]
    lats = [p[1] for p in polygon]
    return {
        "lon_min": min(lons),
        "lon_max": max(lons),
        "lat_min": min(lats),
        "lat_max": max(lats),
    }


def generate_grid_points_in_polygon(polygon, resolution, max_points=None):
    """
    Generate grid coordinate points within polygon bounds
    max_points: Maximum point limit, None means no limit (generates all points within polygon bounds)
    """
    bounds = get_polygon_bounds(polygon)
    points = []

    lat = bounds["lat_min"]
    while lat <= bounds["lat_max"]:
        lon = bounds["lon_min"]
        while lon <= bounds["lon_max"]:
            # Only keep points inside polygon
            if point_in_polygon(lon, lat, polygon):
                points.append((round(lat, 2), round(lon, 2)))
                # Only check limit if max_points is set
                if max_points is not None and len(points) >= max_points:
                    return points
            lon += resolution
        lat += resolution

    return points


def get_api_config():
    """
    Return corresponding API endpoint and names parameter based on DATA_MODE
    """
    if DATA_MODE == "single_year":
        return {
            "url": API_ENDPOINTS["single_year"],
            "names": str(SINGLE_YEAR),  # Single year mode: Use specific year
        }
    else:  # tmy mode
        return {
            "url": API_ENDPOINTS["tmy"],
            "names": "tmy-2024",  # TMY mode: Use tmy-2024
        }


def request_download_link(lat, lon):
    """
    Request data for a single coordinate point
    Uses CSV direct download mode (synchronous), returns CSV content instead of download link
    Supports multiple API Key rotation
    """
    api_config = get_api_config()

    while True:  # Continue retrying until successful
        # Get currently available API Key
        cred = api_key_manager.get_current()
        api_key = cred["api_key"]
        email = cred["email"]
        key_index = cred["index"]

        # CSV direct download uses GET request, parameters in URL
        params = {
            "api_key": api_key,
            "wkt": f"POINT({lon} {lat})",
            "names": api_config["names"],
            "attributes": "air_temperature,clearsky_dhi,clearsky_dni,clearsky_ghi,dhi,dni,ghi,wind_speed",
            "interval": "60",
            "utc": "false",
            "email": email,
        }

        try:
            response = requests.get(
                api_config["url"],
                params=params,
                headers={"x-api-key": api_key},
                timeout=120,  # CSV download may take longer
            )

            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "").lower()
                if "csv" in content_type or "text" in content_type:
                    # Successfully retrieved CSV data
                    api_key_manager.switch_to_next()
                    return response.text, None, key_index + 1
                else:
                    # Returned non-CSV content, possibly an error
                    return None, f"Unexpected content type: {content_type}", key_index + 1

            elif response.status_code in (429, 400):
                # Rate limit or request error
                error_text = response.text[:200] if response.text else str(response.status_code)
                if "email" in error_text.lower():
                    print(f"  ⚠️  API Key #{key_index + 1} email validation failed")
                else:
                    print(f"  ⚠️  API Key #{key_index + 1} encountered HTTP {response.status_code}")
                api_key_manager.mark_cooldown(key_index)
                api_key_manager.switch_to_next(silent=False)
                continue

            else:
                return None, f"HTTP {response.status_code}", key_index + 1

        except Exception as e:
            return None, str(e), key_index + 1


def download_and_extract(item, output_dir, print_lock):
    """
    Real-time detection and download/extraction of single file
    """
    lat, lon, url = item["lat"], item["lon"], item["url"]
    zip_path = output_dir / f"temp_{lat}_{lon}.zip"

    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        if attempt == 1:
            wait_time = DOWNLOAD_INITIAL_WAIT
        else:
            wait_time = DOWNLOAD_RETRY_WAIT

        time.sleep(wait_time)

        try:
            response = requests.get(url, stream=True, timeout=300)

            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "").lower()

                if "zip" in content_type or "octet-stream" in content_type:
                    # Download file
                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    # Extract
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            file_list = zip_ref.namelist()
                            zip_ref.extractall(output_dir)

                        # Delete ZIP
                        zip_path.unlink()

                        csv_count = len([f for f in file_list if f.endswith(".csv")])
                        return True, f"{csv_count} CSV files"

                    except zipfile.BadZipFile:
                        if zip_path.exists():
                            zip_path.unlink()
                        continue
                else:
                    if attempt % 10 == 0:
                        with print_lock:
                            print(
                                f"      [{lat}, {lon}] Waiting for data preparation... (Retry {attempt})"
                            )
                    continue

            elif response.status_code == 404:
                continue
            else:
                continue

        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue

    return False, "Exceeded maximum retries"


def download_worker(download_queue, output_dir, results, print_lock, stop_event):
    """Download worker thread"""
    while not stop_event.is_set():
        try:
            item = download_queue.get(timeout=2)
            if item is None:
                break

            success, msg = download_and_extract(item, output_dir, print_lock)

            with print_lock:
                results["total"] += 1
                if success:
                    results["success"] += 1
                    print(f"  ✓ Download complete ({item['lat']}, {item['lon']}) - {msg}")
                else:
                    results["failed"] += 1
                    results["failed_items"].append(item)
                    print(f"  ✗ Download failed ({item['lat']}, {item['lon']}) - {msg}")

            download_queue.task_done()

        except Exception:
            continue


def save_progress(progress_file, results, pending_count):
    """Save progress"""
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "success": results["success"],
                "failed": results["failed"],
                "pending": pending_count,
                "failed_items": results["failed_items"],
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():
    # Get current mode configuration
    api_config = get_api_config()

    print("=" * 70)
    if DATA_MODE == "single_year":
        print(f"New York State NSRDB PSM Single-Year Data Batch Download (Year: {SINGLE_YEAR})")
        print("  ✓ All points use same year, consistent timeline")
    else:
        print("New York State NSRDB TMY Data Batch Download (Typical Meteorological Year)")
        print("  ⚠️ Note: TMY data years may vary by location")
    print("=" * 70)
    print(f"Data mode: {DATA_MODE.upper()}")
    print(f"API endpoint: {api_config['url'].split('/')[-1]}")
    print(f"names parameter: {api_config['names']}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        f"Max download points: {'No limit (actual bounds)' if MAX_POINTS is None else MAX_POINTS}"
    )
    print(f"Grid resolution: {GRID_RESOLUTION} degrees")
    print(f"Request interval: {REQUEST_INTERVAL} seconds")
    print(f"Parallel download threads: {DOWNLOAD_THREADS}")
    print(f"API Key count: {len(API_CREDENTIALS)} keys")
    for i, cred in enumerate(API_CREDENTIALS):
        masked_key = cred["api_key"][:8] + "..." + cred["api_key"][-4:]
        print(f"  Key#{i+1}: {masked_key} ({cred['email']})")
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress_file = Path(__file__).parent / "batch_progress.json"

    # Generate coordinate points (only points within polygon)
    print("Generating grid coordinate points (New York State polygon bounds only)...")
    bounds = get_polygon_bounds(NY_POLYGON)
    print(
        f"  Polygon bounding box: Longitude [{bounds['lon_min']:.2f}, {bounds['lon_max']:.2f}], Latitude [{bounds['lat_min']:.2f}, {bounds['lat_max']:.2f}]"
    )

    points = generate_grid_points_in_polygon(NY_POLYGON, GRID_RESOLUTION, MAX_POINTS)
    print(f"  Generated {len(points)} valid coordinate points within polygon")
    print()

    # CSV direct download mode: synchronous download, no async queue needed
    results = {"total": 0, "success": 0, "failed": 0, "failed_items": []}

    print("Starting data download (CSV direct mode)...")
    print("-" * 70)

    try:
        for i, (lat, lon) in enumerate(points):
            csv_content, error, used_key = request_download_link(lat, lon)

            results["total"] += 1

            if csv_content:
                # Save CSV file directly
                # Extract location_id from CSV content (second row, second column)
                try:
                    lines = csv_content.strip().split('\n')
                    if len(lines) >= 2:
                        meta_values = lines[1].split(',')
                        location_id = meta_values[1] if len(meta_values) > 1 else f"{lat}_{lon}"
                    else:
                        location_id = f"{lat}_{lon}"
                except Exception:
                    location_id = f"{lat}_{lon}"

                # Generate filename
                year_suffix = api_config["names"]
                filename = f"{location_id}_{lat}_{lon}_{year_suffix}.csv"
                filepath = OUTPUT_DIR / filename

                # Save file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(csv_content)

                results["success"] += 1
                print(f"[{i+1}/{len(points)}] ({lat}, {lon}) [Key#{used_key}] -> {filename}")

            else:
                results["failed"] += 1
                results["failed_items"].append({"lat": lat, "lon": lon, "error": str(error)})
                print(f"[{i+1}/{len(points)}] ({lat}, {lon}) [Key#{used_key}] - Failed: {error}")

            # Save progress every 50 downloads
            if (i + 1) % 50 == 0:
                save_progress(progress_file, results, len(points) - i - 1)

            # API rate limiting
            time.sleep(REQUEST_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal, stopping...")

    # Final progress save
    save_progress(progress_file, results, 0)

    print()
    print("=" * 70)
    print("All complete!")
    print(f"  Success: {results['success']}, Failed: {results['failed']}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    # Count CSV files
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    print(f"\nTotal {len(csv_files)} CSV files in directory")

    # Print failed downloads if any
    if results["failed_items"]:
        print(f"\nFailed downloads ({len(results['failed_items'])} items):")
        for item in results["failed_items"][:10]:
            print(f"  ({item['lat']}, {item['lon']}): {item.get('error', 'unknown')}")
        if len(results["failed_items"]) > 10:
            print(f"  ... and {len(results['failed_items']) - 10} more")


if __name__ == "__main__":
    main()