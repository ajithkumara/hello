import os
import urllib.request
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE_URL = "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(SCRIPT_DIR, "assets", "images")

pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']

if not os.path.exists(TARGET_DIR):
    try:
        os.makedirs(TARGET_DIR)
        print(f"Created directory: {TARGET_DIR}")
    except OSError as e:
        print(f"Error creating directory {TARGET_DIR}: {e}")

print(f"Downloading images to {TARGET_DIR}...")

for p in pieces:
    url = f"{BASE_URL}{p}.png"
    filename = os.path.join(TARGET_DIR, f"{p}.png")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            data = response.read()
            with open(filename, 'wb') as f:
                f.write(data)
            print(f"Downloaded {p}.png")
    except Exception as e:
        print(f"Error downloading {p}.png: {e}")

print("Done.")
