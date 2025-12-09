"""
Batch Download New York State NSRDB TMY Data
Use a polygon to define the boundary, generate grid coordinate points, and request download point by point.
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
# Multi API Key Configuration - Supports rotation of multiple credentials. Register on https://developer.nrel.gov/signup/ (one email address can register for multiple times)
API_CREDENTIALS = [
    {"api_key": "iHTifjC7W8CCCzF0L1bi4s1baYB7IVw2TE9i8egq", "email": "yystoreyyback@gmail.com"},
    {"api_key": "DUluz2aCN67AFqfUF36rQAMfP2uqYTMea38l2atw", "email": "yystoreyyback@gmail.com"},
    # {
    #     "api_key": "2mF0rdaZK8KBu3PVSrNcar1w0chIo9jGifQTH46ksejCPljGSTpixNCiZAFrk1CCnIPBFJIljQSfkLKy",
    #     "email": "yystoreyyback@gmail.com",
    # },
    # {
    #     "api_key": "sdrA1OwYoUkXDff3njDRu624m002LnUU3c9iT1ijUyN9FCqZno8aekJchj9qSM7iVkyNqOH17BGNRmG",
    #     "email": "yystoreyyback@gmail.com",
    # },
    # {
    #     "api_key": "REjrDUW0jWq8vR3PqKzStiHexjRm9Az0UnBBNeS5",
    #     "email": "yystoreyyback@gmail.com",
    # },
    # {"api_key": "tk0uzHlcxiNk93WSLsBkeLuvVKFuuAjDB2QG7qSm", "email": "yystoreyyback@gmail.com"},
    # {"api_key": "Lvp7FQbtoPnOoam28RbdXeHBxz3lZ0L5Q7BUdO7O", "email": "yystoreyyback@gmail.com"},
    # {
    #     "api_key": "2mF0rdaGSTpixNCiZAFrk1CCnIPBFJIljQSfkLKy",
    #     "email": "yystoreyyback@gmail.com",
    # },
    # {
    #     "api_key": "JIN3fdlv1cmYQxLw7sQKmfA3Pwf7NCqnwafqx353",
    #     "email": "yystoreyyback@gmail.com",
    # },
    # {
    #     "api_key": "fTq1aPuILE52gKQbfhFL2rNgbKmMe86QkVACv1qE",
    #     "email": "yystoreyyback@gmail.com",
    # },
]

BASE_URL = (
    "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-tmy-v4-0-0-download.json"
)

# API Key Cooldown Configuration
COOLDOWN_DURATION = 60  # Cooldown time for a single apiKey (s)
ALL_KEYS_COOLDOWN_WAIT = 30  # Wait time when all Keys are in cooldown (s)

# Precise boundary polygon vertices for New York State (Longitude, Latitude) manually with any map tool
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

# Grid Resolution (degrees) - 0.090° ≈ 8.7km，approx 1956 points ( 0.04°,9665pts ; 0.045,7733 ; 0.05,6238 ; 0.055,5174 ; 0.06,4316 ; 0.07,3195 ; 0.08,2410 ; 0.1,1545 ; 0.2,384 )  )
GRID_RESOLUTION = 0.090

# Output Directory
OUTPUT_DIR = Path(__file__).parent.parent / "AnalyzeData" / "data"

# Download Limits
MAX_POINTS = None  # Set to None for no limit, generates all points within the actual polygon
REQUEST_INTERVAL = 2.1  # Request interval (s), API requires 1 per 2 seconds
DOWNLOAD_INITIAL_WAIT = 10  # Initial wait time for first download attempt (s) shorter with multi-key concurrency
DOWNLOAD_RETRY_WAIT = 3  # Retry wait time (s) Fast polling check
MAX_DOWNLOAD_RETRIES = 100  # Maximum number of retries. Increased due to shorter wait time
DOWNLOAD_THREADS = 5  # Number of parallel download threads (Multi-key, higher concurrency)
# ==================================


class ApiKeyManager:
    """
    API Key Rotation Manager
    - Supports rotation of multiple API Keys
    - Automatically switches and marks as cooling down upon encountering 429/400 errors
    - Waits and retries when all Keys are in cooldown
    """

    def __init__(self, credentials):
        self.credentials = credentials
        self.current_index = 0
        self.cooldown_until = {}  # {index: cooldown_end_timestamp}
        self.lock = threading.Lock()

    def get_current(self):
        """Get the current available API Key and email"""
        with self.lock:
            return self._get_available_credential()

    def _get_available_credential(self):
        """Internal method: Get available credential, wait if all are in cooldown"""
        while True:
            now = time.time()
            available_indices = []

            # Find all available Keys
            for i in range(len(self.credentials)):
                cooldown_end = self.cooldown_until.get(i, 0)
                if now >= cooldown_end:
                    available_indices.append(i)

            if available_indices:
                # Prioritize the current index; if unavailable, use the first available one
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
                print(f"\n⚠️  All API Keys are cooling down, waiting {wait_time:.0f} seconds before retrying...")
                self.lock.release()
                time.sleep(wait_time)
                self.lock.acquire()

    def mark_cooldown(self, index):
        """Mark the specified Key as in cooldown status"""
        with self.lock:
            self.cooldown_until[index] = time.time() + COOLDOWN_DURATION
            print(f"  ⏸️  API Key #{index + 1} entering cooldown period ({COOLDOWN_DURATION}s)")

    def switch_to_next(self, silent=True):
        """
        Switch to the next available Key
        silent: Silent mode, do not print switching information (used during normal rotation)
        """
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.credentials)
            if not silent:
                print(f"  🔄 Switching to API Key #{self.current_index + 1}")

    def get_status(self):
        """Get the status of all Keys"""
        with self.lock:
            now = time.time()
            status = []
            for i, cred in enumerate(self.credentials):
                cooldown_end = self.cooldown_until.get(i, 0)
                if now >= cooldown_end:
                    status.append(f"Key#{i+1}: Available")
                else:
                    remaining = int(cooldown_end - now)
                    status.append(f"Key#{i+1}: Cooling down({remaining}s)")
            return ", ".join(status)


# Global API Key Manager instance
api_key_manager = ApiKeyManager(API_CREDENTIALS)


def point_in_polygon(lon, lat, polygon):
    """
   Ray casting algorithm to determine if a point is inside a polygon
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
    """Get the bounding box of the polygon"""
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
    Generate grid coordinate points within the polygon boundary
    max_points: Maximum number of points limit, None means no limit (generates all points within the actual polygon)
    """
    bounds = get_polygon_bounds(polygon)
    points = []

    lat = bounds["lat_min"]
    while lat <= bounds["lat_max"]:
        lon = bounds["lon_min"]
        while lon <= bounds["lon_max"]:
            # Only keep points inside the polygon
            if point_in_polygon(lon, lat, polygon):
                points.append((round(lat, 2), round(lon, 2)))
                # Only check the limit if max_points is set
                if max_points is not None and len(points) >= max_points:
                    return points
            lon += resolution
        lat += resolution

    return points


def request_download_link(lat, lon, year="tmy-2024"):
    """
    Request the download link for a single coordinate point
    Supports multi API Key rotation:
    - Automatically switches to the next Key after each successful request (to distribute quota usage)
    - Marks as cooling down and switches Key upon encountering 429/400 errors
    """
    wkt = f"POINT({lon} {lat})"

    while True:  # Continually retry until successful
        # Get the current available API Key
        cred = api_key_manager.get_current()
        api_key = cred["api_key"]
        email = cred["email"]
        key_index = cred["index"]

        params = {"api_key": api_key}
        post_data = {
            "wkt": wkt,
            "names": year,
            "attributes": "air_temperature,clearsky_dhi,clearsky_dni,clearsky_ghi,dhi,dni,ghi,wind_speed",
            "interval": "60",
            "utc": "false",
            "email": email,
        }

        try:
            response = requests.post(
                BASE_URL,
                params=params,
                data=post_data,
                headers={"x-api-key": api_key},
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()
                if "errors" in result and result["errors"]:
                    error_msg = str(result["errors"])
                    # Check for rate limit related errors
                    if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                        print(
                            f"  ⚠️  API Key #{key_index + 1} encountered rate limit: {error_msg}"
                        )
                        api_key_manager.mark_cooldown(key_index)
                        api_key_manager.switch_to_next(silent=False)  # Show switch on error
                        continue  # Retry with new Key
                    return None, result["errors"], key_index + 1
                outputs = result.get("outputs", {})
                download_url = outputs.get("downloadUrl", "")
                # Switch to the next Key after success as well, to distribute quota usage
                api_key_manager.switch_to_next()
                return download_url, None, key_index + 1  # Return the used Key number

            elif response.status_code in (429, 400):
                # Rate limit or request error, mark as cooldown and switch Key
                print(f"  ⚠️  API Key #{key_index + 1} encountered HTTP {response.status_code}")
                api_key_manager.mark_cooldown(key_index)
                api_key_manager.switch_to_next(silent=False)  # Show switch on error
                continue  # Retry current point with new Key

            else:
                return None, f"HTTP {response.status_code}", key_index + 1

        except Exception as e:
            return None, str(e), key_index + 1


def download_and_extract(item, output_dir, print_lock):
    """
    Real-time check, download, and extract a single file
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

    return False, "Exceeded maximum retry attempts"


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
    print("=" * 70)
    print("New York State NSRDB TMY Data Batch Download (Polygon Boundary + Point-by-Point Request)")
    print("=" * 70)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(
        f"Max Download Points: {'No limit (by actual range)' if MAX_POINTS is None else MAX_POINTS}"
    )
    print(f"Grid Resolution: {GRID_RESOLUTION} degrees")
    print(f"Request Interval: {REQUEST_INTERVAL} seconds")
    print(f"Parallel Download Threads: {DOWNLOAD_THREADS}")
    print(f"API Key Count: {len(API_CREDENTIALS)} keys")
    for i, cred in enumerate(API_CREDENTIALS):
        masked_key = cred["api_key"][:8] + "..." + cred["api_key"][-4:]
        print(f"  Key#{i+1}: {masked_key} ({cred['email']})")
    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress_file = Path(__file__).parent / "batch_progress.json"

    # Generate coordinate points (only points within the polygon)
    print("Generating grid coordinate points (only within New York State polygon)...")
    bounds = get_polygon_bounds(NY_POLYGON)
    print(
        f"  Boundary Box: Longitude [{bounds['lon_min']:.2f}, {bounds['lon_max']:.2f}], Latitude [{bounds['lat_min']:.2f}, {bounds['lat_max']:.2f}]"
    )

    points = generate_grid_points_in_polygon(NY_POLYGON, GRID_RESOLUTION, MAX_POINTS)
    print(f"  Total {len(points)} valid coordinate points generated within the polygon")
    print()

    # Create download queue and results summary
    download_queue = Queue()
    results = {"total": 0, "success": 0, "failed": 0, "failed_items": []}
    print_lock = threading.Lock()
    stop_event = threading.Event()

    # Start download threads
    download_threads = []
    for i in range(DOWNLOAD_THREADS):
        t = threading.Thread(
            target=download_worker,
            args=(download_queue, OUTPUT_DIR, results, print_lock, stop_event),
            name=f"Downloader-{i+1}",
        )
        t.daemon = True
        t.start()
        download_threads.append(t)

    # Get download links and add to queue in real-time
    print("Starting to get download links (download begins immediately after acquisition)...")
    print("-" * 70)

    link_success = 0
    link_failed = 0

    try:
        for i, (lat, lon) in enumerate(points):
            url, error, used_key = request_download_link(lat, lon)

            with print_lock:
                if url:
                    link_success += 1
                    print(f"[{i+1}/{len(points)}] ({lat}, {lon}) [Key#{used_key}]")
                    print(f"    Link: {url}")

                    # Add immediately to the download queue
                    download_queue.put({"lat": lat, "lon": lon, "url": url})
                else:
                    link_failed += 1
                    print(
                        f"[{i+1}/{len(points)}] ({lat}, {lon}) [Key#{used_key}] - Failed to get link: {error}"
                    )

            # Save progress every 50 points
            if (i + 1) % 50 == 0:
                save_progress(progress_file, results, download_queue.qsize())

            # API Rate Limit
            time.sleep(REQUEST_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nInterrupt signal received, stopping...")
        stop_event.set()

    print()
    print("-" * 70)
    print(f"Link acquisition complete: Success {link_success}, Failed {link_failed}")
    print()

    # Wait for all downloads to finish
    print("Waiting for all download tasks to complete...")
    print(f"Remaining in queue: {download_queue.qsize()} tasks")

    download_queue.join()

    # Stop download threads
    stop_event.set()
    for _ in download_threads:
        download_queue.put(None)
    for t in download_threads:
        t.join(timeout=5)

    # Final progress save
    save_progress(progress_file, results, 0)

    print()
    print("=" * 70)
    print("All finished!")
    print(f"  Link Acquisition: Success {link_success}, Failed {link_failed}")
    print(f"  File Download: Success {results['success']}, Failed {results['failed']}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print("=" * 70)

    # Count CSV files
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    print(f"\nThere are a total of {len(csv_files)} CSV files in the directory")

    # Print failed items, if any
    if results["failed_items"]:
        print(f"\nFailed Downloads ({len(results['failed_items'])} items):")
        for item in results["failed_items"][:10]:
            print(f"  ({item['lat']}, {item['lon']})")
        if len(results["failed_items"]) > 10:
            print(f"  ... and {len(results['failed_items']) - 10} more")


if __name__ == "__main__":
    main()
