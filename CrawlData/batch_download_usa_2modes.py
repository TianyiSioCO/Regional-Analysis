"""
Batch download NSRDB solar radiation data for the Contiguous United States (CONUS)
Supports two modes:
  1. TMY mode: Download Typical Meteorological Year data (source years may vary)
  2. PSM Single Year mode: Download actual data for a specific year (consistent timeline for all points)

Uses a polygon to define boundaries, generates grid coordinate points, and requests downloads point by point.
Boundary data source: https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA.geo.json
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
# Data Mode Selection
# - "tmy": Download TMY Typical Meteorological Year data (Default, but source years may vary)
# - "single_year": Download PSM data for a specific year (Recommended, consistent timeline)
DATA_MODE = "single_year"  # Options: "tmy" or "single_year"

# Year setting for Single Year mode (Effective when DATA_MODE="single_year")
# Available range: 1998-2023 (CONUS)
SINGLE_YEAR = 2020  # Recommended to choose a year with normal meteorological conditions

# API Endpoint Configuration
# Note: Use .csv direct download mode to avoid 404 issues with asynchronous ZIP mode
API_ENDPOINTS = {
    "tmy": "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-tmy-v4-0-0-download.csv",
    "single_year": "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv",
}

# Multi-API Key Configuration - Supports rotation of multiple credentials
# Note: NREL API email validation is strict; use meaningful email formats
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

# API Key Cooldown Configuration
COOLDOWN_DURATION = 60  # Cooldown time for a single Key (seconds)
ALL_KEYS_COOLDOWN_WAIT = 30  # Wait time when all Keys are cooling down (seconds)

# Simplified polygon vertices for the Contiguous United States (CONUS) (Longitude, Latitude)
# Data source: https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA.geo.json
# Longitude range: approx -124.69 to -66.96
# Latitude range: approx 25.08 to 49.39
US_MAINLAND_POLYGON = [
    (-94.81758, 49.38905),
    (-94.64, 48.84),
    (-94.32914, 48.67074),
    (-93.63087, 48.60926),
    (-92.61, 48.45),
    (-91.64, 48.14),
    (-90.83, 48.27),
    (-89.6, 48.01),
    (-89.272917, 48.019808),
    (-88.378114, 48.302918),
    (-87.439793, 47.94),
    (-86.461991, 47.553338),
    (-85.652363, 47.220219),
    (-84.87608, 46.900083),
    (-84.779238, 46.637102),
    (-84.543749, 46.538684),
    (-84.6049, 46.4396),
    (-84.3367, 46.40877),
    (-84.14212, 46.512226),
    (-84.091851, 46.275419),
    (-83.890765, 46.116927),
    (-83.616131, 46.116927),
    (-83.469551, 45.994686),
    (-83.592851, 45.816894),
    (-82.550925, 45.347517),
    (-82.337763, 44.44),
    (-82.137642, 43.571088),
    (-82.43, 42.98),
    (-82.9, 42.43),
    (-83.12, 42.08),
    (-83.142, 41.975681),
    (-83.02981, 41.832796),
    (-82.690089, 41.675105),
    (-82.439278, 41.675105),
    (-81.277747, 42.209026),
    (-80.247448, 42.3662),
    (-78.939362, 42.863611),
    (-78.92, 42.965),
    (-79.01, 43.27),
    (-79.171674, 43.466339),
    (-78.72028, 43.625089),
    (-77.737885, 43.629056),
    (-76.820034, 43.628784),
    (-76.5, 44.018459),
    (-76.375, 44.09631),
    (-75.31821, 44.81645),
    (-74.867, 45.00048),
    (-73.34783, 45.00738),
    (-71.50506, 45.0082),
    (-71.405, 45.255),
    (-71.08482, 45.30524),
    (-70.66, 45.46),
    (-70.305, 45.915),
    (-69.99997, 46.69307),
    (-69.237216, 47.447781),
    (-68.905, 47.185),
    (-68.23444, 47.35486),
    (-67.79046, 47.06636),
    (-67.79134, 45.70281),
    (-67.13741, 45.13753),
    (-66.96466, 44.8097),
    (-68.03252, 44.3252),
    (-69.06, 43.98),
    (-70.11617, 43.68405),
    (-70.645476, 43.090238),
    (-70.81489, 42.8653),
    (-70.825, 42.335),
    (-70.495, 41.805),
    (-70.08, 41.78),
    (-70.185, 42.145),
    (-69.88497, 41.92283),
    (-69.96503, 41.63717),
    (-70.64, 41.475),
    (-71.12039, 41.49445),
    (-71.86, 41.32),
    (-72.295, 41.27),
    (-72.87643, 41.22065),
    (-73.71, 40.931102),
    (-72.24126, 41.11948),
    (-71.945, 40.93),
    (-73.345, 40.63),
    (-73.982, 40.628),
    (-73.952325, 40.75075),
    (-74.25671, 40.47351),
    (-73.96244, 40.42763),
    (-74.17838, 39.70926),
    (-74.90604, 38.93954),
    (-74.98041, 39.1964),
    (-75.20002, 39.24845),
    (-75.52805, 39.4985),
    (-75.32, 38.96),
    (-75.071835, 38.782032),
    (-75.05673, 38.40412),
    (-75.37747, 38.01551),
    (-75.94023, 37.21689),
    (-76.03127, 37.2566),
    (-75.72205, 37.93705),
    (-76.23287, 38.319215),
    (-76.35, 39.15),
    (-76.542725, 38.717615),
    (-76.32933, 38.08326),
    (-76.989998, 38.239992),
    (-76.30162, 37.917945),
    (-76.25874, 36.9664),
    (-75.9718, 36.89726),
    (-75.86804, 36.55125),
    (-75.72749, 35.55074),
    (-76.36318, 34.80854),
    (-77.397635, 34.51201),
    (-78.05496, 33.92547),
    (-78.55435, 33.86133),
    (-79.06067, 33.49395),
    (-79.20357, 33.15839),
    (-80.301325, 32.509355),
    (-80.86498, 32.0333),
    (-81.33629, 31.44049),
    (-81.49042, 30.72999),
    (-81.31371, 30.03552),
    (-80.98, 29.18),
    (-80.535585, 28.47213),
    (-80.53, 28.04),
    (-80.056539, 26.88),
    (-80.088015, 26.205765),
    (-80.13156, 25.816775),
    (-80.38103, 25.20616),
    (-80.68, 25.08),
    (-81.17213, 25.20126),
    (-81.33, 25.64),
    (-81.71, 25.87),
    (-82.24, 26.73),
    (-82.70515, 27.49504),
    (-82.85526, 27.88624),
    (-82.65, 28.55),
    (-82.93, 29.1),
    (-83.70959, 29.93656),
    (-84.1, 30.09),
    (-85.10882, 29.63615),
    (-85.28784, 29.68612),
    (-85.7731, 30.15261),
    (-86.4, 30.4),
    (-87.53036, 30.27433),
    (-88.41782, 30.3849),
    (-89.18049, 30.31598),
    (-89.593831, 30.159994),
    (-89.413735, 29.89419),
    (-89.43, 29.48864),
    (-89.21767, 29.29108),
    (-89.40823, 29.15961),
    (-89.77928, 29.30714),
    (-90.15463, 29.11743),
    (-90.880225, 29.148535),
    (-91.626785, 29.677),
    (-92.49906, 29.5523),
    (-93.22637, 29.78375),
    (-93.84842, 29.71363),
    (-94.69, 29.48),
    (-95.60026, 28.73863),
    (-96.59404, 28.30748),
    (-97.14, 27.83),
    (-97.37, 27.38),
    (-97.38, 26.69),
    (-97.33, 26.21),
    (-97.14, 25.87),
    (-97.53, 25.84),
    (-98.24, 26.06),
    (-99.02, 26.37),
    (-99.3, 26.84),
    (-99.52, 27.54),
    (-100.11, 28.11),
    (-100.45584, 28.69612),
    (-100.9576, 29.38071),
    (-101.6624, 29.7793),
    (-102.48, 29.76),
    (-103.11, 28.97),
    (-103.94, 29.27),
    (-104.45697, 29.57196),
    (-104.70575, 30.12173),
    (-105.03737, 30.64402),
    (-105.63159, 31.08383),
    (-106.1429, 31.39995),
    (-106.50759, 31.75452),
    (-108.24, 31.754854),
    (-108.24194, 31.34222),
    (-109.035, 31.34194),
    (-111.02361, 31.33472),
    (-113.30498, 32.03914),
    (-114.815, 32.52528),
    (-114.72139, 32.72083),
    (-115.99135, 32.61239),
    (-117.12776, 32.53534),
    (-117.295938, 33.046225),
    (-117.944, 33.621236),
    (-118.410602, 33.740909),
    (-118.519895, 34.027782),
    (-119.081, 34.078),
    (-119.438841, 34.348477),
    (-120.36778, 34.44711),
    (-120.62286, 34.60855),
    (-120.74433, 35.15686),
    (-121.71457, 36.16153),
    (-122.54747, 37.55176),
    (-122.51201, 37.78339),
    (-122.95319, 38.11371),
    (-123.7272, 38.95166),
    (-123.86517, 39.76699),
    (-124.39807, 40.3132),
    (-124.17886, 41.14202),
    (-124.2137, 41.99964),
    (-124.53284, 42.76599),
    (-124.14214, 43.70838),
    (-124.020535, 44.615895),
    (-123.89893, 45.52341),
    (-124.079635, 46.86475),
    (-124.39567, 47.72017),
    (-124.68721, 48.184433),
    (-124.566101, 48.379715),
    (-123.12, 48.04),
    (-122.58736, 47.096),
    (-122.34, 47.36),
    (-122.5, 48.18),
    (-122.84, 49),
    (-120, 49),
    (-117.03121, 49),
    (-116.04818, 49),
    (-113, 49),
    (-110.05, 49),
    (-107.05, 49),
    (-104.04826, 48.99986),
    (-100.65, 49),
    (-97.22872, 49.0007),
    (-95.15907, 49),
    (-95.15609, 49.38425),
    (-94.81758, 49.38905),  # Closed polygon
]



# Grid Resolution (degrees) - 0.095° ≈ 9.2km, approx 1700 points
GRID_RESOLUTION = 0.9

# Output Directory
OUTPUT_DIR = Path(__file__).parent.parent / "AnalyzeData" / "data"

# Download Limits
MAX_POINTS = None  # Set to None for no limit, generates all points within polygon range
REQUEST_INTERVAL = 2.1  # Request interval (seconds), API requires 1 per 2 seconds
DOWNLOAD_INITIAL_WAIT = 10  # Initial wait time for download (seconds)
DOWNLOAD_RETRY_WAIT = 3  # Retry wait time (seconds)
MAX_DOWNLOAD_RETRIES = 100  # Max retries
DOWNLOAD_THREADS = 10  # Parallel download threads
# ==================================


class ApiKeyManager:
    """
    API Key Rotation Manager
    - Supports rotation of multiple API Keys
    - Automatically switches and marks cooldown on 429/400 errors
    - Waits if all Keys are in cooldown
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
        """Internal method: Get available credential, wait if all are cooling down"""
        while True:
            now = time.time()
            available_indices = []

            # Find all available Keys
            for i in range(len(self.credentials)):
                cooldown_end = self.cooldown_until.get(i, 0)
                if now >= cooldown_end:
                    available_indices.append(i)

            if available_indices:
                # Prioritize current index, otherwise use first available
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
                # All Keys are in cooldown, wait and retry
                min_wait = min(self.cooldown_until.values()) - now
                wait_time = max(min_wait, ALL_KEYS_COOLDOWN_WAIT)
                print(f"\n⚠️  All API Keys are in cooldown, waiting {wait_time:.0f} seconds to retry...")
                self.lock.release()
                time.sleep(wait_time)
                self.lock.acquire()

    def mark_cooldown(self, index):
        """Mark specific Key as cooling down"""
        with self.lock:
            self.cooldown_until[index] = time.time() + COOLDOWN_DURATION
            print(f"  ⏸️  API Key #{index + 1} entered cooldown ({COOLDOWN_DURATION}s)")

    def switch_to_next(self, silent=True):
        """
        Switch to the next available Key
        silent: If true, suppresses switching info (used for normal rotation)
        """
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.credentials)
            if not silent:
                print(f"  🔄 Switched to API Key #{self.current_index + 1}")

    def get_status(self):
        """Get status of all Keys"""
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


# Global instance of API Key Manager
api_key_manager = ApiKeyManager(API_CREDENTIALS)


def point_in_polygon(lon, lat, polygon):
    """
    Ray-casting algorithm to determine if a point is inside a polygon
    polygon: list of [(lon1, lat1), (lon2, lat2), ...]
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
    """Get bounding box of a polygon"""
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
    Generate grid coordinate points within polygon range
    max_points: Max points limit, None means no limit
    """
    bounds = get_polygon_bounds(polygon)
    points = []

    lat = bounds["lat_min"]
    while lat <= bounds["lat_max"]:
        lon = bounds["lon_min"]
        while lon <= bounds["lon_max"]:
            # Keep only points inside the polygon
            if point_in_polygon(lon, lat, polygon):
                points.append((round(lat, 2), round(lon, 2)))
                # Check limit only if max_points is set
                if max_points is not None and len(points) >= max_points:
                    return points
            lon += resolution
        lat += resolution

    return points


def get_api_config():
    """
    Returns API endpoint and names parameters based on DATA_MODE
    """
    if DATA_MODE == "single_year":
        return {
            "url": API_ENDPOINTS["single_year"],
            "names": str(SINGLE_YEAR),  # Single year mode: use specific year
        }
    else:  # tmy mode
        return {
            "url": API_ENDPOINTS["tmy"],
            "names": "tmy-2024",  # TMY mode: use tmy-2024
        }


def request_download_link(lat, lon):
    """
    Request data for a single coordinate point
    Uses CSV direct download mode (synchronous), returns CSV content
    Supports multi-API Key rotation
    """
    api_config = get_api_config()

    while True:  # Continue retrying until success
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
                timeout=120,  # CSV download might take longer
            )

            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "").lower()
                # Accept csv, text, octet-stream etc.
                if "csv" in content_type or "text" in content_type or "octet-stream" in content_type:
                    try:
                        if "octet-stream" in content_type:
                            content = response.content.decode("utf-8")
                        else:
                            content = response.text

                        # Validate valid CSV format
                        if content and ("," in content or "\n" in content):
                            api_key_manager.switch_to_next()
                            return content, None, key_index + 1
                        else:
                            return None, f"Invalid CSV content", key_index + 1
                    except UnicodeDecodeError:
                        return None, f"Cannot decode content as UTF-8", key_index + 1
                else:
                    return None, f"Unexpected content type: {content_type}", key_index + 1

            elif response.status_code in (429, 400):
                # Rate limit or request error
                error_text = response.text[:200] if response.text else str(response.status_code)
                if "email" in error_text.lower():
                    print(f"  ⚠️  API Key #{key_index + 1} Email validation failed")
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
    Real-time detection and download extraction for a single file
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
                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    try:
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            file_list = zip_ref.namelist()
                            zip_ref.extractall(output_dir)

                        zip_path.unlink()

                        csv_count = len([f for f in file_list if f.endswith(".csv")])
                        return True, f"{csv_count} CSVs"

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

    return False, "Exceeded max retries"


def download_worker(download_queue, output_dir, results, print_lock, stop_event):
    """Download working thread"""
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
        print(f"NSRDB PSM Single Year Batch Download for CONUS (Year: {SINGLE_YEAR})")
        print("  ✓ All points use the same year, timeline consistent")
    else:
        print("NSRDB TMY Batch Download for CONUS (Typical Meteorological Year)")
        print("  ⚠️ Note: TMY source years may vary by point")
    print("=" * 70)
    print(f"Data Mode: {DATA_MODE.upper()}")
    print(f"API Endpoint: {api_config['url'].split('/')[-1]}")
    print(f"Names Parameter: {api_config['names']}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(
        f"Max Points: {'Unlimited (Actual Range)' if MAX_POINTS is None else MAX_POINTS}"
    )
    print(f"Grid Resolution: {GRID_RESOLUTION} degrees")
    print(f"Request Interval: {REQUEST_INTERVAL} seconds")
    print(f"Parallel Download Threads: {DOWNLOAD_THREADS}")
    print(f"API Key Count: {len(API_CREDENTIALS)} units")
    for i, cred in enumerate(API_CREDENTIALS):
        masked_key = cred["api_key"][:8] + "..." + cred["api_key"][-4:]
        print(f"  Key#{i+1}: {masked_key} ({cred['email']})")
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress_file = Path(__file__).parent / "batch_progress.json"

    # Generate points (only inside polygon)
    print("Generating grid points (Contiguous US polygon range only)...")
    bounds = get_polygon_bounds(US_MAINLAND_POLYGON)
    print(
        f"  Polygon BBox: Lon [{bounds['lon_min']:.2f}, {bounds['lon_max']:.2f}], Lat [{bounds['lat_min']:.2f}, {bounds['lat_max']:.2f}]"
    )

    points = generate_grid_points_in_polygon(US_MAINLAND_POLYGON, GRID_RESOLUTION, MAX_POINTS)
    print(f"  Generated {len(points)} valid points within polygon")
    print()

    # CSV direct mode: synchronous download, no async queue needed
    results = {"total": 0, "success": 0, "failed": 0, "failed_items": []}

    print("Starting data download (CSV Direct Mode)...")
    print("-" * 70)

    try:
        for i, (lat, lon) in enumerate(points):
            csv_content, error, used_key = request_download_link(lat, lon)

            results["total"] += 1

            if csv_content:
                # Save CSV file directly
                # Extract location_id (row 2, column 2)
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

            # Save progress every 50 points
            if (i + 1) % 50 == 0:
                save_progress(progress_file, results, len(points) - i - 1)

            # API Throttling
            time.sleep(REQUEST_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nInterrupt received, stopping...")

    # Final progress save
    save_progress(progress_file, results, 0)

    print()
    print("=" * 70)
    print("Task Completed!")
    print(f"  Success: {results['success']}, Failed: {results['failed']}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print("=" * 70)

    # Count CSV files
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    print(f"\nTotal CSV files in directory: {len(csv_files)}")

    # Print failed items if any
    if results["failed_items"]:
        print(f"\nFailed Downloads ({len(results['failed_items'])} items):")
        for item in results["failed_items"][:10]:
            print(f"  ({item['lat']}, {item['lon']}): {item.get('error', 'unknown')}")
        if len(results["failed_items"]) > 10:
            print(f"  ... and {len(results['failed_items']) - 10} more")


if __name__ == "__main__":
    main()