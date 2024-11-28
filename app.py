from flask import Flask, render_template, request, redirect, send_file, url_for
import os
import cv2
import numpy as np
import uuid
import torch
import pathlib
from pathlib import Path
#pathlib.PosixPath = pathlib.WindowsPath


# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv5 model from PyTorch Hub (replace with your trained model path)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt', force_reload=True)

@app.route('/')
def index():
    return render_template('index.html', filename=None)

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect('/')

    video = request.files['video']
    if video.filename == '':
        return redirect('/')

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # Generate a unique filename for the processed video
    output_filename = f'output_{uuid.uuid4().hex}.mp4'
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    process_video(video_path, output_path)

    return render_template('index.html', filename=output_filename)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using mp4v codec for .mp4 output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference using YOLOv5
        results = model(frame)
        detection_frame = np.squeeze(results.render())  # Draw bounding boxes on the frame

        # Write the processed frame to the output video
        out.write(detection_frame)

    cap.release()
    out.release()

@app.route('/outputs/<filename>')
def serve_video(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype='video/mp4')

if __name__ == "__main__":
    app.run(debug=True)
