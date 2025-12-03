from flask import Flask, request, Response, send_from_directory
from gtts import gTTS
import io
import os

app = Flask(__name__)

# Route 1: Serve the main page
@app.route("/")
def home():
    return send_from_directory(".", "tts.html")  # reads tts.html from current dir

# Route 2: TTS API
@app.route("/tts")
def tts():
    text = request.args.get("text", "").strip()
    if not text:
        return {"error": "No text provided"}, 400

    try:
        print(f"✅ Processing TTS for: '{text[:30]}...'")  # Logs in Render
        mp3_fp = io.BytesIO()

        tts = gTTS(text=text[:100], lang="en", slow=False)  # ⚠️ Cap at 100 chars!
        tts.write_to_fp(mp3_fp)
        mp3_data = mp3_fp.getvalue()

        print(f"✅ gTTS succeeded. MP3 size: {len(mp3_data)} bytes")

        if len(mp3_data) < 100:
            print("❌ Warning: MP3 too small — likely failed silently!")
            return {"error": "TTS returned empty audio"}, 500

        return Response(
            mp3_data,
            mimetype="audio/mpeg",
            headers={
                "Content-Length": str(len(mp3_data)),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
            }
        )

    except Exception as e:
        import traceback
        print("❌ TTS EXCEPTION:")
        traceback.print_exc()  # Full stack trace in logs
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import os
    # Render sets PORT automatically (default: 10000)
    port = int(os.environ.get("PORT", 5000))
    # MUST bind to 0.0.0.0 — not 127.0.0.1!
    app.run(host="0.0.0.0", port=port, debug=False)