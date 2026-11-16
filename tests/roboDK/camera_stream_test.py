import io
import os
import subprocess
import numpy as np
from PIL import Image
from robodk.robolink import Robolink, ITEM_TYPE_CAMERA

# 1. Multi-IP fallback connection
def get_candidate_ips():
    candidates = []
    try:
        ns_ip = subprocess.getoutput("grep nameserver /etc/resolv.conf | awk '{print $2}'").strip()
        if ns_ip and not ns_ip.startswith("127."):
            candidates.append(ns_ip)
    except Exception:
        pass
    candidates.append("127.0.0.1")
    return list(dict.fromkeys(candidates))

RDK = None
for ip in get_candidate_ips():
    try:
        test_rdk = Robolink(robodk_ip=ip, port=20500)
        if test_rdk.Connect():
            print(f"Connected successfully via {ip}!")
            RDK = test_rdk
            break
    except Exception:
        pass

if RDK is None or not RDK.Connect():
    print("[ERROR] Could not connect to RoboDK.")
    exit(1)

# 2. Reference the camera item
cam_item = RDK.Item('Camera 1', ITEM_TYPE_CAMERA)
if not cam_item.Valid():
    cam_item = RDK.Item('Camera', ITEM_TYPE_CAMERA)

if not cam_item.Valid():
    print("[ERROR] Could not find camera item in RoboDK.")
    exit(1)

print(f"Streaming from: {cam_item.Name()}")

# 3. Pull image directly into memory over socket (empty string "")
bytes_img = RDK.Cam2D_Snapshot('', cam_item)

if not bytes_img or not isinstance(bytes_img, bytes):
    print("[ERROR] Did not receive image bytes from RoboDK.")
    exit(1)

# Decode image bytes directly in RAM
pil_image = Image.open(io.BytesIO(bytes_img))
img_array = np.array(pil_image)

print("\n--- Image Successfully Captured in Memory ---")
print(f"Array Shape (Height, Width, Channels): {img_array.shape}")
print(f"Data Type: {img_array.dtype}")
print(f"Pixel Range: [{img_array.min()}, {img_array.max()}]")

# 4. Save locally in WSL Linux to verify visually
output_path = os.path.expanduser("~/tbp.monty/tests/roboDK/test_capture.png")
pil_image.save(output_path)
print(f"Saved local verification copy to: {output_path}")