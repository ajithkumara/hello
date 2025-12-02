from flask import Flask, request, send_file, jsonify
from gtts import gTTS
import soundfile as sf
import io

app = Flask(__name__)

@app.route("/tts")
def tts():
    text = request.args.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Step 1: Generate MP3 in memory
        mp3_bytes = io.BytesIO()
        tts = gTTS(text=text, lang="en")
        tts.write_to_fp(mp3_bytes)
        mp3_bytes.seek(0)

        # Step 2: Convert MP3 -> WAV
        data, samplerate = sf.read(mp3_bytes, dtype="int16")
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, data, samplerate, format="WAV")
        wav_bytes.seek(0)

        return send_file(
            wav_bytes,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="speech.wav"
        )

    except Exception as e:
        print("TTS ERROR:", e)
        return jsonify({"error": str(e)}), 500
