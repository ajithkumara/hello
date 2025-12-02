from flask import Flask, render_template, request, send_file
from gtts import gTTS
import soundfile as sf
import os

app = Flask(__name__, static_folder='.', static_url_path='')


# -------------------------
# HOME PAGE (index.html in root)
# -------------------------
@app.route('/')
def home():
    return app.send_static_file('index.html')


# -------------------------
# TEXT TO SPEECH PAGE
# -------------------------
@app.route('/tts')
def tts_page():
    return render_template("tts.html")


@app.route('/tts/generate', methods=['POST'])
def generate_tts():
    text = request.form.get("text")
    if not text:
        return "No text provided"

    filename = "output.wav"

    # Generate speech
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

    # Fix WAV format using soundfile
    data, samplerate = sf.read(filename)
    sf.write(filename, data, samplerate)

    return send_file(filename, as_attachment=True)


# -------------------------
# RUN APP
# -------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
