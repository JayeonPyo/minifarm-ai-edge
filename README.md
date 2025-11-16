# MiniFarm AI Edge Node

[![oneM2M](https://img.shields.io/badge/oneM2M-TR--0071-blue)](https://www.onem2m.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)


AI-powered plant health monitoring system with oneM2M TR-0071 compliant dynamic model deployment architecture.

---

### oneM2M Resource Structure (TR-0071)

```
/oneM2M_CSE/
├── modelRepo/                    # <mlModel> resources
│   ├── mlModel-species/
│   │   └── (CIN) {"mlModelPath": "/path/best.pt"}
│   └── mlModel-health/
│       └── (CIN) {"mlModelPath": "/path/last.pt"}
│
└── SodaFarm/ (AE)
    ├── modelDeploymentList/      # <modelDeployment> resources
    │   ├── modelDeploy_species/
    │   └── modelDeploy_healthy/
    │
    ├── inference/                # <inferenceOutput> resources
    │   ├── species/              # Detection results
    │   └── health/
    │
    ├── Sensors/                  # Sensor data
    └── Actuators/                # Control commands
```

---

## Quick Start

### Prerequisites

- **Hardware**: Raspberry Pi 4 (4GB+ RAM), Pi Camera Module
- **Software**: Python 3.9+, TinyIoT Server (oneM2M CSE)

### Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/minifarm-ai-edge.git
cd minifarm-ai-edge
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup oneM2M resources**
```bash
python src/setup_resources.py
```
This creates TR-0071 compliant resource structure on TinyIoT server.

4. **Configure settings**
Edit `src/minifarm_ai_main.py`:
```python
TINYIOT_URL = "http://YOUR_SERVER_IP:3000"
RASPI_ORIGIN = "SRaspberryPi_AI"
TARGET_HOURS = [8, 13, 18]  # Analysis schedule
```

5. **Run AI edge node**
```bash
python src/minifarm_ai_main.py
```

### Test Mode
```bash
# Run immediate detection (bypass schedule)
python src/minifarm_ai_main.py test_camera
```
---

## How It Works

### 1. Dynamic Configuration Loading

The AI edge node loads model paths dynamically from oneM2M server:

```python
# Fetch deployment config
deploy_data = get_latest_cin_json(
    f"{TINYIOT_URL}/TinyIoT/TinyFarm/modelDeploymentList/modelDeploy_healthy/la"
)

# Get model metadata
model_id = deploy_data["modelID"]
model_data = get_latest_cin_json(f"{TINYIOT_URL}{model_id}/la")

# Load YOLO model (no hardcoded path!)
model = YOLO(model_data["mlModelPath"])
```

**Key Benefit**: Update model path on server → All edge nodes auto-apply on next boot (0s maintenance)

### 2. Dual-Phase Detection

Single YOLOv8 model outputs: `healthy_basil`, `unhealthy_poinsettia`, etc.

Post-processing splits into:
- **species_data**: Unique species list (no duplicates)
- **health_data**: All detected plants with health status

### 3. Real-time Streaming

- **High-res stream** (4608×3456): AI analysis only (3 times/day)
- **Low-res stream** (640×480): 24/7 MJPEG streaming 

---

## File Structure

```
src/
├── minifarm_ai_main.py          # Main AI edge node
│   ├── _load_config_from_cse()  # Dynamic config loading
│   ├── detect()                 # YOLOv8 inference + post-processing
│   ├── _send_to_tinyiot()       # Upload results to oneM2M
│   └── start_streaming_server() # MJPEG streaming thread
│
└── setup_resources.py           # oneM2M resource initialization
    ├── create_ae()              # Create Application Entity
    ├── create_container()       # Create containers (CNT)
    ├── create_content_instance()# Store data (CIN)
    └── create_or_update_acp()   # Access control policy
```

---


## Acknowledgments

- [oneM2M](https://www.onem2m.org/) for TR-0071 standards
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8



