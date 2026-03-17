import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FLOOD_FILE = os.path.join(DATA_DIR, "Flood.csv")
RUNOFF_FILE = os.path.join(DATA_DIR, "Runoff.csv")

FLOOD_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "flood_routing")
RUNOFF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "runoff_routing")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FLOOD_OUTPUT_DIR, exist_ok=True)
os.makedirs(RUNOFF_OUTPUT_DIR, exist_ok=True)