from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import time
import torch
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model #type: ignore
from torchvision import transforms
from flask_cors import CORS
import logging
import datetime
import uuid
import json

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Configure logging

logger = logging.getLogger(__name__)

# App configuration
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "mp4", "mov"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULTS_FOLDER"] = RESULTS_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Device configuration for deepfake detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the trained model
try:
    model_path = os.environ.get("MODEL_PATH", "models/xception_deepfake_image_5o.h5")
    model = load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None


def preprocess_image(image, target_size=(299, 299)):
    """Preprocess image for model input"""
    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, target_size)
        preprocessed = tf.keras.applications.xception.preprocess_input(resized)
        return np.expand_dims(preprocessed, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


def predict(frame):
    """Run prediction on a single frame"""
    try:
        preprocessed = preprocess_image(frame)
        prediction = model.predict(preprocessed)[0][0]
        label = "FAKE" if prediction >= 0.5 else "REAL"
        confidence = float(prediction) if prediction >= 0.5 else float(1 - prediction)
        return label, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


def detect_image(image_path):
    """Detect deepfakes in an image"""
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return "ERROR", 0.0

        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Could not read image: {image_path}")
            return "ERROR", 0.0

        label, confidence = predict(frame)

        # Create visualization for results
        output_path = os.path.join(
            app.config["RESULTS_FOLDER"], f"{os.path.basename(image_path)}"
        )
        visualization = frame.copy()

        # Add text label
        text = f"{label} ({confidence*100:.2f}%)"
        color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
        cv2.putText(
            visualization, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
        )

        # Save visualization
        cv2.imwrite(output_path, visualization)

        logger.info(
            f"Image detection completed: {image_path} -> {label} ({confidence:.4f})"
        )
        return label, confidence
    except Exception as e:
        logger.error(f"Error in detect_image: {str(e)}")
        return "ERROR", 0.0


def detect_video(video_path, thorough_mode=False):
    """Detect deepfakes in a video with frame sampling"""
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video not found: {video_path}")
            return "ERROR", 0.0

        # Generate a unique output filename
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(
            app.config["RESULTS_FOLDER"], f"{name}_analyzed{ext}"
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return "ERROR", 0.0

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Set frame sampling rate based on mode
        frame_sample_rate = (
            1 if thorough_mode else max(1, int(fps / 2))
        )  # Every frame or 2 frames per second

        frame_count = 0
        fake_frames = 0
        real_frames = 0
        total_confidence = 0

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames according to sample rate
            if frame_count % frame_sample_rate == 0:
                label, confidence = predict(frame)
                total_confidence += confidence

                if label == "FAKE":
                    fake_frames += 1
                else:
                    real_frames += 1

                # Add labeled visualization
                text = f"{label} ({confidence*100:.2f}%)"
                color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
                cv2.putText(
                    frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
                )

            # Write the frame to output video
            out.write(frame)
            frame_count += 1

            # Progress logging for long videos
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

        # Release resources
        cap.release()
        out.release()

        # Determine overall verdict
        processed_frames = fake_frames + real_frames
        if processed_frames == 0:
            logger.error("No frames were processed successfully")
            return "ERROR", 0.0

        fake_ratio = fake_frames / processed_frames
        overall_label = (
            "FAKE" if fake_ratio > 0.3 else "REAL"
        )  # If more than 30% frames are fake, mark video as fake
        avg_confidence = total_confidence / processed_frames

        logger.info(
            f"Video detection completed: {video_path} -> {overall_label} ({avg_confidence:.4f})"
        )
        return overall_label, avg_confidence

    except Exception as e:
        logger.error(f"Error in detect_video: {str(e)}")
        return "ERROR", 0.0


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_media(filepath, thorough_mode=False):
    """Analyze the uploaded media file (image or video)"""
    start_time = time.time()
    logger.info(f"Starting analysis of {filepath} with thorough_mode={thorough_mode}")

    try:
        if filepath.lower().endswith((".mp4", ".mov")):
            # Video file
            result, confidence = detect_video(filepath, thorough_mode)
        else:
            # Image file
            result, confidence = detect_image(filepath)

        analysis_time = time.time() - start_time

        response = {
            "result": result,
            "confidence": confidence,
            "analysis_time": f"{analysis_time:.2f} seconds",
            "file_type": (
                "video" if filepath.lower().endswith((".mp4", ".mov")) else "image"
            ),
            "timestamp": datetime.datetime.now().isoformat(),
            "unique_id": str(uuid.uuid4()),
        }

        # Save analysis results for history tracking
        result_log_path = os.path.join(
            app.config["RESULTS_FOLDER"], "analysis_history.json"
        )
        try:
            if os.path.exists(result_log_path):
                with open(result_log_path, "r") as f:
                    history = json.load(f)
            else:
                history = []

            # Add filename to response for history
            response_with_filename = response.copy()
            response_with_filename["filename"] = os.path.basename(filepath)

            history.append(response_with_filename)

            with open(result_log_path, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving to analysis history: {str(e)}")

        logger.info(
            f"Analysis completed: {result} with {confidence:.4f} confidence in {analysis_time:.2f}s"
        )
        return response

    except Exception as e:
        logger.error(f"Error in analyze_media: {str(e)}")
        analysis_time = time.time() - start_time
        return {
            "result": "ERROR",
            "confidence": 0.0,
            "analysis_time": f"{analysis_time:.2f} seconds",
            "error": str(e),
        }


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect")
def detect_page():
    return render_template("detect.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how-it-works.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/api/detect", methods=["POST"])
def detect_deepfake_api():
    """API endpoint for deepfake detection"""
    try:
        # Check if file is present
        if "file" not in request.files:
            logger.warning("No file provided in request")
            return jsonify({"error": "No file provided", "status": 400}), 400

        file = request.files["file"]

        # Check if file is empty
        if file.filename == "":
            logger.warning("Empty file name in request")
            return jsonify({"error": "No selected file", "status": 400}), 400

        # Check file format
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file format: {file.filename}")
            return (
                jsonify(
                    {
                        "error": "Invalid file format. Supported formats: JPG, PNG, MP4, MOV",
                        "status": 400,
                    }
                ),
                400,
            )

        # Generate a unique filename to prevent overwriting
        original_filename = secure_filename(file.filename)
        unique_filename = f"{int(time.time())}_{original_filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        # Save file
        file.save(filepath)
        logger.info(f"File saved to {filepath}")

        # Get detection mode
        thorough_mode = request.form.get("mode", "standard").lower() == "thorough"

        # Analyze the media
        analysis_result = analyze_media(filepath, thorough_mode)

        return jsonify(analysis_result)

    except Exception as e:
        logger.error(f"Error in detect_deepfake_api: {str(e)}")
        return (
            jsonify(
                {"error": "Internal server error", "details": str(e), "status": 500}
            ),
            500,
        )


@app.route("/detect", methods=["POST"])
def detect_deepfake():
    """Form submission endpoint for deepfake detection"""
    try:
        # Check if file is present
        if "file" not in request.files:
            logger.warning("No file provided in request")
            return jsonify({"error": "No file provided", "status": 400}), 400

        file = request.files["file"]

        # Check if file is empty
        if file.filename == "":
            logger.warning("Empty file name in request")
            return jsonify({"error": "No selected file", "status": 400}), 400

        # Check file format
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file format: {file.filename}")
            return (
                jsonify(
                    {
                        "error": "Invalid file format. Supported formats: JPG, PNG, MP4, MOV",
                        "status": 400,
                    }
                ),
                400,
            )

        # Generate a unique filename to prevent overwriting
        original_filename = secure_filename(file.filename)
        unique_filename = f"{int(time.time())}_{original_filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

        # Save file
        file.save(filepath)

        # Get detection mode
        thorough_mode = request.form.get("mode", "standard").lower() == "thorough"

        # Analyze the media
        analysis_result = analyze_media(filepath, thorough_mode)

        return jsonify(analysis_result)

    except Exception as e:
        logger.error(f"Error in detect_deepfake: {str(e)}")
        return (
            jsonify(
                {"error": "Internal server error", "details": str(e), "status": 500}
            ),
            500,
        )


@app.route("/api/history", methods=["GET"])
def get_analysis_history():
    """API endpoint to retrieve analysis history"""
    try:
        result_log_path = os.path.join(
            app.config["RESULTS_FOLDER"], "analysis_history.json"
        )

        if not os.path.exists(result_log_path):
            return jsonify([])

        with open(result_log_path, "r") as f:
            history = json.load(f)

        return jsonify(history)

    except Exception as e:
        logger.error(f"Error in get_analysis_history: {str(e)}")
        return (
            jsonify(
                {"error": "Internal server error", "details": str(e), "status": 500}
            ),
            500,
        )


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.datetime.now().isoformat(),
                "model_loaded": model is not None,
            }
        )
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV", "production") == "development"

    logger.info(f"Starting DeepGuard server on port {port}, debug={debug_mode}")
    app.run(host="0.0.0.0", port=port, debug=False)
