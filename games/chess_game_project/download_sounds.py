import os
import urllib.request
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Using Lichess standard sounds (OGG is better for Pygame Sound objects)
BASE_URL = "https://raw.githubusercontent.com/lichess-org/lila/master/public/sound/standard/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(SCRIPT_DIR, "assets", "sounds")

sounds = {
    'move': 'Move.ogg',
    'capture': 'Capture.mp3', # Capture might only be mp3 in some themes, let's try both
    'check': 'Check.ogg'
}

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)
    print(f"Created directory: {TARGET_DIR}")

print(f"Downloading sounds to {TARGET_DIR}...")

for name, filename in sounds.items():
    # Try OGG first, then MP3
    success = False
    for ext in ['ogg', 'mp3']:
        actual_filename = f"{filename.split('.')[0]}.{ext}"
        url = f"{BASE_URL}{actual_filename}"
        target_path = os.path.join(TARGET_DIR, f"{name}.{ext}")
        try:
            print(f"Attempting to download {actual_filename}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                data = response.read()
                with open(target_path, 'wb') as f:
                    f.write(data)
                print(f"Successfully downloaded {name}.{ext}")
                success = True
                break
        except Exception as e:
            continue
    
    if not success:
        print(f"Failed to download {name} in any format.")

print("\nDone! I've updated the script to prefer .ogg files which work better in Pygame.")
