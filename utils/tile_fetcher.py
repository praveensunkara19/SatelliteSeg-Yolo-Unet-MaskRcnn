# tile_fetcher.py

import math
import requests
from PIL import Image
from io import BytesIO

def deg2num(lat_deg, lon_deg, zoom):
    """Convert latitude and longitude to tile x, y coordinates."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def fetch_tile_from_coords(xtile, ytile, zoom):
    """Fetch a single tile from Esri by tile x, y coordinates."""
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def fetch_stitched_map(lat, lon, zoom, num_tiles=3):
    """
    Fetch a stitched Esri satellite image centered at (lat, lon).
    Returns a PIL.Image stitched from num_tiles x num_tiles grid.
    """
    center_xtile, center_ytile = deg2num(lat, lon, zoom)
    half = num_tiles // 2
    tile_size = 256  # Standard Web Mercator tile size
    stitched_image = Image.new("RGB", (tile_size * num_tiles, tile_size * num_tiles))

    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            xtile = center_xtile + dx
            ytile = center_ytile + dy

            try:
                tile = fetch_tile_from_coords(xtile, ytile, zoom)
                stitched_image.paste(
                    tile,
                    ((dx + half) * tile_size, (dy + half) * tile_size)
                )
            except Exception as e:
                print(f"Skipping tile ({xtile}, {ytile}) due to error: {e}")

    return stitched_image
