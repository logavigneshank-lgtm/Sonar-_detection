import os
from pathlib import Path
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

CLASS_NAMES = ["aircraft", "fish", "other", "shipwreck"]
MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
model = YOLO(MODEL_PATH)

HTML = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sonar Marine Detection</title>
<style>
body{margin:0;font-family:Arial,sans-serif;background:#071827;color:#fff}
.wrap{max-width:900px;margin:auto;padding:28px 18px}
.card{background:#10283c;border-radius:18px;padding:24px;box-shadow:0 10px 35px #0006}
h1{margin:0 0 8px;font-size:28px} .sub{opacity:.8}
form{margin-top:22px;border:2px dashed #55738a;border-radius:14px;padding:25px;text-align:center}
input{max-width:100%;padding:10px} button{padding:12px 22px;margin-top:14px;border:0;border-radius:9px;font-weight:bold;cursor:pointer}
.result{margin-top:22px;background:#0b1e2e;padding:18px;border-radius:12px}
.item{padding:12px 0;border-bottom:1px solid #ffffff22}
.item:last-child{border-bottom:0}
img{max-width:100%;margin-top:15px;border-radius:10px}
.err{margin-top:18px;padding:12px;border-radius:9px;background:#4b1f25}
.badge{display:inline-block;padding:5px 9px;border-radius:999px;background:#183e56}
</style>
</head>
<body>
<div class="wrap"><div class="card">
<h1>AI-Powered Underwater Marine Debris & Anomaly Detection</h1>
<div class="sub">Side-Scan Sonar Object Detection</div>
<form method="post" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required>
<br><button type="submit">Detect Object</button>
</form>
{% if error %}<div class="err">{{error}}</div>{% endif %}
{% if results %}
<div class="result"><h2>Detection Results</h2>
{% for r in results %}
<div class="item"><span class="badge">{{r.label}}</span> — {{ "%.1f"|format(r.confidence*100) }}% confidence</div>
{% endfor %}
</div>
{% endif %}
</div></div>
</body></html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    results, error = [], None
    if request.method == "POST":
        f = request.files.get("image")
        if not f or not f.filename:
            error = "Please select an image."
        else:
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            path = upload_dir / secure_filename(f.filename)
            f.save(path)

            predictions = model.predict(
                source=str(path),
                conf=0.50,
                iou=0.45,
                verbose=False
            )
            for box in predictions[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if 0 <= cls_id < len(CLASS_NAMES):
                    results.append({"label": CLASS_NAMES[cls_id], "confidence": conf})

            if not results:
                error = "No confident object detected."
    return render_template_string(HTML, results=results, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
