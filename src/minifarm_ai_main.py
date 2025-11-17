#!/usr/bin/env python3
"""
MiniFarm AI Detector - TR-0071 Compliant Dynamic Model Deployment

This module implements an AI edge node for smart farming with:
- Dynamic model configuration loading from oneM2M server
- YOLOv8-based plant health detection
- Real-time MJPEG streaming
- Scheduled detection at specific hours
"""

# Libs for streaming and threading
import io
import logging
import socketserver
from http import server
from threading import Condition, Thread
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# Standard libs
import requests
import json
import time
import os
from datetime import datetime
from ultralytics import YOLO
import cv2
import numpy as np

# ==================== Configuration (Defaults) ====================
TINYIOT_URL = "http://YOUR_SERVER_IP:3000"
AE_NAME = "TinyFarm"
RASPI_ORIGIN = "SRaspberryPi_AI"
SAVE_DIR = "/tmp/minifarm_captures"
os.makedirs(SAVE_DIR, exist_ok=True)
STREAMING_PORT = 5000

# AI Detection Timing Configuration
TARGET_HOURS = list(range(24))
CHECK_INTERVAL_SECONDS = 60 * 10  # Check every 10 minutes

# ====================  Streaming Server Classes ====================
class StreamingOutput(io.BufferedIOBase):
    """Output buffer for camera frames."""
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    """HTTP handler for the MJPEG stream."""
    output_instance = None

    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = f"""
            <html><head><title>MiniFarm Live Stream</title></head>
            <body>
                <h1 style="font-family: sans-serif;">MiniFarm Live Stream</h1>
                <img src="stream.mjpg" width="640" height="480" />
            </body></html>
            """.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                if not StreamingHandler.output_instance:
                    raise Exception("Streaming output not set")
                
                while True:
                    with StreamingHandler.output_instance.condition:
                        StreamingHandler.output_instance.condition.wait()
                        frame = StreamingHandler.output_instance.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

# ==================== Helper: oneM2M GET ====================
def get_latest_cin_json(container_url, origin):
    """Gets the latest CIN from a container and parses its JSON content."""
    url = f"{container_url}/la"
    headers = {
        'X-M2M-Origin': origin, 'X-M2M-RVI': '2a', 'Accept': 'application/json',
    }
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            cin_data = r.json().get("m2m:cin", {})
            json_content = cin_data.get("con")
            
            if not json_content:
                print(f"[GET ERROR] {url} - 'con' is empty")
                return None

            # Check if content is already a dict
            if isinstance(json_content, dict):
                print(f"[GET SUCCESS] {url}")
                return json_content
            
            # If string, parse it
            print(f"[GET SUCCESS] {url}")
            return json.loads(json_content)
            
        elif r.status_code == 403:
            print(f"[GET FAIL] {url} - HTTP 403 (Forbidden). Check ACP.")
            return None
        elif r.status_code == 404:
            print(f"[GET FAIL] {url} - HTTP 404 (Not Found).")
            return None
        else:
            print(f"[GET FAIL] {url} - HTTP {r.status_code}")
            return None
    except requests.exceptions.Timeout:
        print(f"[GET EXCEPTION] {url} - Timeout")
        return None
    except json.JSONDecodeError as e:
        print(f"[GET EXCEPTION] {url} - JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"[GET EXCEPTION] {url} - {e}")
        return None

# ==================== Main AI Class ====================

class MiniFarmAI:
    def __init__(self):
        print("=" * 60)
        print("MiniFarm AI Initialize - TR-0071")
        print("=" * 60)
        
        self.config = self._load_config_from_cse()
        
        if not self.config:
            print("FATAL ERROR: Failed to load config from CSE.")
            print("Please run 'setup_resources.py' first.")
            raise Exception("CSE Config Load Failed")

        print("\n[CSE Config Loaded]")
        print(f"  Model (Health): {self.config['model_health_path']}")
        print(f"  Output (Species): {self.config['output_species']}")
        print(f"  Output (Health): {self.config['output_health']}")

        self._init_camera()
        self._init_models()
        
        self.http_server = None
        self.start_streaming_server()
        
        self.last_detection_hour = -1
        
        print("\nInitialize Complete!\n")
    
    def _load_config_from_cse(self):
        """Loads configuration from oneM2M CSE."""
        print("Loading configuration from oneM2M CSE...")
        deploy_list_path = f"{TINYIOT_URL}/TinyIoT/{AE_NAME}/modelDeploymentList"
        config = {}
        try:
            # Get Species deployment config
            deploy_species_url = f"{deploy_list_path}/modelDeploy_species"
            deploy_species_data = get_latest_cin_json(deploy_species_url, RASPI_ORIGIN)
            if not deploy_species_data:
                return None

            # Get Health deployment config
            deploy_health_url = f"{deploy_list_path}/modelDeploy_healthy"
            deploy_health_data = get_latest_cin_json(deploy_health_url, RASPI_ORIGIN)
            if not deploy_health_data:
                return None

            # Get model metadata
            model_species_id_path = deploy_species_data.get("modelID")
            model_health_id_path = deploy_health_data.get("modelID")
            
            model_species_data = get_latest_cin_json(f"{TINYIOT_URL}{model_species_id_path}", RASPI_ORIGIN)
            model_health_data = get_latest_cin_json(f"{TINYIOT_URL}{model_health_id_path}", RASPI_ORIGIN)
            
            if not model_species_data or not model_health_data:
                return None

            # Store configuration
            config = {
                "model_species_path": model_species_data.get("mlModelPath"),
                "model_health_path": model_health_data.get("mlModelPath"),
                "output_species": deploy_species_data.get("outputResource"),
                "output_health": deploy_health_data.get("outputResource"),
                "url_species": f"{TINYIOT_URL}{deploy_species_data.get('outputResource')}",
                "url_health": f"{TINYIOT_URL}{deploy_health_data.get('outputResource')}"
            }
            
            return config
            
        except Exception as e:
            print(f"[ERROR] Config loading failed: {e}")
            return None

    def _init_camera(self):
        """Initialize Raspberry Pi Camera with dual streams."""
        print("\n[Camera Init]")
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(
            main={"size": (4608, 3456)},  # High-res for AI
            lores={"size": (640, 480)},   # Low-res for streaming
            buffer_count=2
        )
        self.camera.configure(config)
        self.camera.start()
        print("Camera started: Main(4608x3456), Lores(640x480)")

    def _init_models(self):
        """Load YOLO models."""
        print("\n[Model Init]")
        print(f"Loading model: {self.config['model_health_path']}")
        self.model_main = YOLO(self.config['model_health_path'])
        print("Model loaded successfully")

    def start_streaming_server(self):
        """Start MJPEG streaming server in separate thread."""
        print("\n[Streaming Server Init]")
        self.output = StreamingOutput()
        self.camera.start_encoder(JpegEncoder(), FileOutput(self.output), name="lores")
        StreamingHandler.output_instance = self.output

        def run_server():
            address = ('', STREAMING_PORT)
            self.http_server = socketserver.ThreadingTCPServer(address, StreamingHandler)
            self.http_server.serve_forever()

        thread = Thread(target=run_server, daemon=True)
        thread.start()
        print(f"Streaming server started on port {STREAMING_PORT}")

    def extract_species_from_label(self, label):
        """Extract species name from health label."""
        label_lower = label.lower()
        if "basil" in label_lower or "bail" in label_lower:
            return "basil"
        elif "poin" in label_lower:
            return "poinsettia"
        return "unknown"

    def detect(self):
        """Run AI detection and send results to oneM2M."""
        print("\n" + "="*60)
        print("Starting AI Detection")
        print("="*60)
        
        try:
            # Capture high-res image
            timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
            raw_path = os.path.join(SAVE_DIR, f"raw_{timestamp_file}.jpg")
            
            frame_bgr = self.camera.capture_array("main")
            cv2.imwrite(raw_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"[CAPTURE] Saved: {raw_path}")
            
            # Run inference
            print("\n[INFERENCE] Running YOLOv8...")
            results_health = self.model_main(raw_path, conf=0.5, verbose=False)
            
            # Process results
            health_data = {}
            unique_species_set = set()
            health_boxes = []
            
            detections_health = results_health[0].boxes
            print(f"[INFERENCE] Detected {len(detections_health)} objects")
            
            for i in range(len(detections_health)):
                box = detections_health.xyxy[i].cpu().numpy().astype(int)
                health_label = results_health[0].names[int(detections_health.cls[i])]
                conf = float(detections_health.conf[i])
                
                # Fix typos
                if "poinsenttia" in health_label:
                    health_label = health_label.replace("poinsenttia", "poinsettia")
                
                health_data[str(i)] = health_label
                health_boxes.append((box, health_label, conf))
                
                species = self.extract_species_from_label(health_label)
                if species != "unknown":
                    unique_species_set.add(species)
                
                print(f"  [{i}] {health_label} (conf: {conf:.2f})")
            
            # Create species data
            species_data = {str(idx): species for idx, species in enumerate(sorted(list(unique_species_set)))}
            
            print(f"\n[RESULT] Species: {species_data}")
            print(f"[RESULT] Health: {health_data}")
            
            # Visualize
            frame_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated_frame = frame_display.copy()
            
            for box, health_label, conf in health_boxes:
                x1, y1, x2, y2 = box
                color = (0, 255, 0) if "healthy" in health_label.lower() else (255, 0, 0)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                label = f"{health_label} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save annotated
            detection_path = os.path.join(SAVE_DIR, f"detection_{timestamp_file}.jpg")
            cv2.imwrite(detection_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR), 
                       [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"[SAVE] Detection saved: {detection_path}")
            
            # Send to TinyIoT
            success = self._send_to_tinyiot(species_data, health_data)
            return success
            
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _send_to_tinyiot(self, species_data, health_data):
        """Send detection results to oneM2M server."""
        print("\n[UPLOAD] Sending to TinyIoT...")
        
        url_species = self.config['url_species']
        url_health = self.config['url_health']
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        headers = {
            'X-M2M-Origin': RASPI_ORIGIN, 
            'X-M2M-RVI': '2a', 
            'Accept': 'application/json',
            'Content-Type': 'application/json; ty=4', 
            'X-M2M-RI': 'create_cin_' + str(time.time()),
        }
        all_success = True
        
        # Send species
        if species_data:
            species_with_timestamp = {"timestamp": timestamp, "data": species_data}
            json_string = json.dumps(species_with_timestamp)
            payload = {"m2m:cin": {"con": "data", "lbl": [json_string]}}
            try:
                r = requests.post(url_species, headers=headers, json=payload, timeout=10)
                if r.status_code == 201:
                    print(f"[SUCCESS] Species uploaded")
                else:
                    print(f"[FAIL] Species upload failed: HTTP {r.status_code}")
                    all_success = False
            except Exception as e:
                print(f"[FAIL] Species upload error: {e}")
                all_success = False
        
        # Send health
        if health_data:
            health_with_timestamp = {"timestamp": timestamp, "data": health_data}
            json_string = json.dumps(health_with_timestamp)
            payload = {"m2m:cin": {"con": "data", "lbl": [json_string]}}
            try:
                headers_health = headers.copy()
                headers_health['X-M2M-RI'] = 'create_cin_' + str(time.time() + 1)
                r = requests.post(url_health, headers=headers_health, json=payload, timeout=10)
                if r.status_code == 201:
                    print(f"[SUCCESS] Health uploaded")
                else:
                    print(f"[FAIL] Health upload failed: HTTP {r.status_code}")
                    all_success = False
            except Exception as e:
                print(f"[FAIL] Health upload error: {e}")
                all_success = False
        
        if all_success:
            print("[UPLOAD] All data sent successfully")
        return all_success
    
    def run(self):
        """Main loop: Check time and run detection at TARGET_HOURS."""
        print("\n" + "="*60)
        print("Real-time Monitoring Started")
        print(f"AI Detection Schedule: {TARGET_HOURS}")
        print(f"Check Interval: {CHECK_INTERVAL_SECONDS}s")
        print("="*60 + "\n")
        
        try:
            while True:
                now = datetime.now()
                current_hour = now.hour

                if current_hour in TARGET_HOURS and current_hour != self.last_detection_hour:
                    print(f"\n[Target Hour {current_hour}:00] Running AI Detection...")
                    self.detect()
                    self.last_detection_hour = current_hour
                    print(f"\nDetection complete. Waiting for next check...")

                else:
                    if current_hour != self.last_detection_hour:
                        self.last_detection_hour = -1
                    
                    print(f"Time: {now.strftime('%H:%M:%S')}. Next check in {CHECK_INTERVAL_SECONDS}s...", end='\r')
                
                time.sleep(CHECK_INTERVAL_SECONDS)
                
        except KeyboardInterrupt:
            print("\n\nProgram Terminated (Ctrl+C)")
        
        finally:
            print("Stopping camera...")
            self.camera.stop_encoder()
            self.camera.stop()
            
            if self.http_server:
                print("Stopping streaming server...")
                self.http_server.shutdown()
                self.http_server.server_close()
            
            print("Exit complete.")


# ==================== Main ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test_camera":
            print("\n=== Camera Test Mode ===")
            ai = None
            try:
                ai = MiniFarmAI()
                print("\n[Test] Running immediate detection...")
                ai.detect()
                print("\n[Test] Complete.")
            except Exception as e:
                print(f"\n[Test] Failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if ai:
                    ai.camera.stop_encoder()
                    ai.camera.stop()
                    if ai.http_server:
                        ai.http_server.shutdown()
                        ai.http_server.server_close()
    else:
        # Normal operation
        ai = MiniFarmAI()
        ai.run()
