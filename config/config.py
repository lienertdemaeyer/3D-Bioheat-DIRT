import os

# Paths
BASE_DIR = r"C:\Users\liene\OneDrive - Universiteit Antwerpen\InViLab\Code\Bioheat_Perforator_Mapping"
H5_DIR = r"C:\Users\liene\Documents\H5 - Output"
COCO_PATH = r"C:\Users\liene\Documents\Data - Thermography\20250225_Patients_CoolingTime\TIFF\QuPath Annotations Cooled_area\GeoJSON\coco_annotations_cooled_area.json"
OUTPUT_DIR = r"C:\Users\liene\Documents\DATA OUTPUT\3D-BIOHEAT-DIRT"

# Parameters
PIXEL_SIZE_MM = 0.5
MAX_FRAMES = 100

# Bioheat Parameters
RHO = 1050.0
C = 3600.0
K = 0.4
ALPHA = (K / (RHO * C)) * 0.3
SMOOTHING_SIGMA = 2.0

# Window for Analysis (Frames)
START_FRAME = 15
END_FRAME = 45

# Patients to Process (P15-P25)
PATIENTS = []
for p in range(15, 26):
    for m in range(1, 5):
        PATIENTS.append(f"P{p:02d}M{m:02d}")

