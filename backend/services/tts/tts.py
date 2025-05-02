from flask import Flask, request, send_file
from gtts import gTTS
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get("text")
    lang = data.get("lang", "te")  # Telugu

    # Generate audio in memory
    tts = gTTS(text=text, lang=lang)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    return send_file(mp3_fp, mimetype='audio/mpeg')

if __name__ == '__main__':
    app.run(port=5003)
