#!/usr/bin/env python3
"""
oneM2M TR-0071 Resource Setup Script

This script creates the TR-0071 compliant resource structure on a TinyIoT server
for dynamic AI model deployment in smart farming applications.
"""

import requests
import json

# ==================== Configuration ====================
TINYIOT_URL = "http://YOUR_SERVER_IP:3000"
AE_NAME = "TinyFarm"
CSE_BASE_PATH = f"{TINYIOT_URL}/TinyIoT"

REPO_PATH = f"{CSE_BASE_PATH}/modelRepo"
AE_PATH = f"{CSE_BASE_PATH}/{AE_NAME}"
DEPLOY_LIST_PATH = f"{AE_PATH}/modelDeploymentList"
INFERENCE_PATH = f"{AE_PATH}/inference"

ADMIN_ORIGIN = "SRaspberryPi_AI"
ACP_NAME = "acp_pi_full_access"

# ==================== Data Definitions ====================
ML_MODEL_SPECIES_DATA = {
    "name": "species-detector",
    "version": "v1.0",
    "platform": "Ultralytics/PyTorch",
    "description": "Plant species detection (basil, poinsettia)",
    "mlModelURL": "",
    "mlModelPath": "/home/seslab/minifarm/best.pt"
}

ML_MODEL_HEALTH_DATA = {
    "name": "health-species-detector",
    "version": "v2.1",
    "platform": "Ultralytics/PyTorch",
    "description": "Health+Species detection (healthy_basil, etc.)",
    "mlModelURL": "",
    "mlModelPath": "/home/seslab/minifarm/last.pt"
}

DEPLOY_SPECIES_DATA = {
    "modelID": "/TinyIoT/modelRepo/mlModel-species",
    "inputResource": f"/TinyIoT/{AE_NAME}/Camera/Images",
    "outputResource": f"/TinyIoT/{AE_NAME}/inference/species",
    "modelStatus": "deployed"
}

DEPLOY_HEALTHY_DATA = {
    "modelID": "/TinyIoT/modelRepo/mlModel-health",
    "inputResource": f"/TinyIoT/{AE_NAME}/Camera/Images",
    "outputResource": f"/TinyIoT/{AE_NAME}/inference/health",
    "modelStatus": "deployed"
}

# ==================== Helper Functions ====================

def create_ae(parent_url, ae_name, origin):
    """Creates an Application Entity (AE) resource (ty=2)."""
    headers = {
        'X-M2M-Origin': origin,
        'X-M2M-RVI': '2a',
        'Accept': 'application/json',
        'Content-Type': 'application/json; ty=2',
        'X-M2M-RI': f"create_ae_{ae_name}",
    }
    payload = {
        "m2m:ae": {
            "rn": ae_name,
            "api": "N.org.example.my-app",
            "rr": True
        }
    }
    
    try:
        r = requests.post(parent_url, headers=headers, json=payload, timeout=5)
        if r.status_code == 201:
            print(f"[SUCCESS] AE '{ae_name}' created")
            return True
        elif r.status_code == 409:
            print(f"[INFO] AE '{ae_name}' already exists")
            return True
        else:
            print(f"[FAIL] Failed to create AE '{ae_name}': HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Exception creating AE '{ae_name}': {e}")
        return False

def create_or_update_acp(parent_url, acp_name, origin_id, origin):
    """Creates or retrieves Access Control Policy (ACP) and returns resource ID."""
    print(f"\n[ACP] Checking/Creating ACP '{acp_name}'...")
    check_headers = {
        'X-M2M-Origin': origin,
        'X-M2M-RVI': '2a',
        'Accept': 'application/json'
    }
    
    # Check if ACP already exists
    try:
        check_url = f"{parent_url}/{acp_name}"
        r_check = requests.get(check_url, headers=check_headers, timeout=5)
        if r_check.status_code == 200:
            ri = r_check.json().get("m2m:acp", {}).get("ri")
            print(f"[INFO] ACP '{acp_name}' already exists, ri: {ri}")
            return ri
    except:
        pass
    
    # Create new ACP
    headers = {
        'X-M2M-Origin': origin,
        'X-M2M-RVI': '2a',
        'Accept': 'application/json',
        'Content-Type': 'application/json; ty=1',
        'X-M2M-RI': f"create_acp_{acp_name}",
    }
    
    # acop: 63 = All permissions (Create, Retrieve, Update, Delete, Notify, Discover)
    access_rule = {"acop": 63, "acor": [origin_id]}
    payload = {
        "m2m:acp": {
            "rn": acp_name,
            "pv": {"acr": [access_rule]},
            "pvs": {"acr": [access_rule]}
        }
    }
    
    try:
        r = requests.post(parent_url, headers=headers, json=payload, timeout=5)
        if r.status_code == 201:
            ri = r.json().get("m2m:acp", {}).get("ri")
            print(f"[SUCCESS] ACP '{acp_name}' created, ri: {ri}")
            return ri
        elif r.status_code == 409:
            r_get = requests.get(f"{parent_url}/{acp_name}", headers=check_headers, timeout=5)
            if r_get.status_code == 200:
                ri = r_get.json().get("m2m:acp", {}).get("ri")
                print(f"[INFO] ACP '{acp_name}' already exists, ri: {ri}")
                return ri
        else:
            print(f"[FAIL] Failed to create ACP: HTTP {r.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] Exception creating ACP: {e}")
        return None

def create_container(parent_url, container_name, origin, acp_ri_list=None):
    """Creates a Container (CNT) resource (ty=3)."""
    headers = {
        'X-M2M-Origin': origin,
        'X-M2M-RVI': '2a',
        'Accept': 'application/json',
        'Content-Type': 'application/json; ty=3',
        'X-M2M-RI': f"create_cnt_{container_name}",
    }
    
    container_payload = {"rn": container_name}
    if acp_ri_list:
        container_payload["acpi"] = acp_ri_list
        print(f"[ACP] Linking ACP to '{container_name}'")
    
    payload = {"m2m:cnt": container_payload}
    
    try:
        r = requests.post(parent_url, headers=headers, json=payload, timeout=5)
        if r.status_code == 201:
            print(f"[SUCCESS] Container '{container_name}' created")
            return True
        elif r.status_code == 409:
            print(f"[INFO] Container '{container_name}' already exists")
            return True
        else:
            print(f"[FAIL] Failed to create container: HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Exception creating container: {e}")
        return False

def create_content_instance(container_url, data_dict, origin):
    """Creates a Content Instance (CIN) with JSON content (ty=4)."""
    headers = {
        'X-M2M-Origin': origin,
        'X-M2M-RVI': '2a',
        'Accept': 'application/json',
        'Content-Type': 'application/json; ty=4',
        'X-M2M-RI': f"create_cin_{container_url.split('/')[-1]}",
    }
    
    json_string = json.dumps(data_dict)
    payload = {
        "m2m:cin": {
            "con": json_string,
            "lbl": ["config", data_dict.get("name", "deployment")]
        }
    }
    
    try:
        r = requests.post(container_url, headers=headers, json=payload, timeout=5)
        if r.status_code == 201:
            print(f"[SUCCESS] CIN created")
            print(f"  Data: {json_string[:60]}...")
            return True
        else:
            print(f"[FAIL] Failed to create CIN: HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"[ERROR] Exception creating CIN: {e}")
        return False

# ==================== Main Execution ====================
def main():
    print("=" * 60)
    print("oneM2M TR-0071 Resource Setup")
    print(f"Target CSE: {TINYIOT_URL}")
    print(f"AE Name: {AE_NAME}")
    print(f"Originator: {ADMIN_ORIGIN}")
    print("=" * 60)

    # Step 0: Create AE
    print("\n[Step 0] Creating Application Entity...")
    if not create_ae(CSE_BASE_PATH, AE_NAME, ADMIN_ORIGIN):
        print("Failed to create AE. Stopping.")
        return
    
    # Step 1: Create ACP
    acp_ri = create_or_update_acp(CSE_BASE_PATH, ACP_NAME, ADMIN_ORIGIN, ADMIN_ORIGIN)
    if not acp_ri:
        print("Failed to create ACP. Stopping.")
        return
    
    acp_ri_list = [acp_ri]
    print(f"[ACP] Using ACP ID: {acp_ri_list}")

    # Step 2: Create modelDeploymentList container
    print(f"\n[Step 1] Creating modelDeploymentList...")
    if not create_container(AE_PATH, "modelDeploymentList", ADMIN_ORIGIN, acp_ri_list):
        print("Failed to create modelDeploymentList. Stopping.")
        return

    # Step 3: Create model metadata in modelRepo
    print("\n[Step 2] Setting up modelRepo...")
    if create_container(REPO_PATH, "mlModel-species", ADMIN_ORIGIN, acp_ri_list):
        create_content_instance(f"{REPO_PATH}/mlModel-species", ML_MODEL_SPECIES_DATA, ADMIN_ORIGIN)
    
    if create_container(REPO_PATH, "mlModel-health", ADMIN_ORIGIN, acp_ri_list):
        create_content_instance(f"{REPO_PATH}/mlModel-health", ML_MODEL_HEALTH_DATA, ADMIN_ORIGIN)

    # Step 4: Create deployment configurations
    print(f"\n[Step 3] Creating deployment configurations...")
    if create_container(DEPLOY_LIST_PATH, "modelDeploy_species", ADMIN_ORIGIN, acp_ri_list):
        create_content_instance(f"{DEPLOY_LIST_PATH}/modelDeploy_species", DEPLOY_SPECIES_DATA, ADMIN_ORIGIN)
    
    if create_container(DEPLOY_LIST_PATH, "modelDeploy_healthy", ADMIN_ORIGIN, acp_ri_list):
        create_content_instance(f"{DEPLOY_LIST_PATH}/modelDeploy_healthy", DEPLOY_HEALTHY_DATA, ADMIN_ORIGIN)

    # Step 5: Create inference output containers
    print(f"\n[Step 4] Creating inference output containers...")
    if create_container(AE_PATH, "inference", ADMIN_ORIGIN, acp_ri_list):
        create_container(INFERENCE_PATH, "species", ADMIN_ORIGIN, acp_ri_list)
        create_container(INFERENCE_PATH, "health", ADMIN_ORIGIN, acp_ri_list)

    print("\n" + "=" * 60)
    print("Resource Setup Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify resources at: " + TINYIOT_URL)
    print("2. Update model paths in mlModel-* if needed")
    print("3. Run minifarm_ai_main.py on Raspberry Pi")

if __name__ == "__main__":
    main()
