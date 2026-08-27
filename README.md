# Final Sonar Marine Debris Detector

Uses the trained YOLO model `best.pt`.

Classes:
- aircraft
- fish
- other
- shipwreck

The web application uses confidence threshold 0.50 to suppress weak false detections.

## Local
pip install -r requirements.txt
python app.py

## Render
Build: pip install -r requirements.txt
Start: gunicorn app:app
