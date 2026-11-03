from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUTS / "models"

RANDOM_SEED = 42
DATASET_NAME = "DEEL-AI/NOTAM"
MAX_TOKEN_LENGTH = 128
NUM_CLASSES = 13
VAL_SPLIT_FRACTION = 0.2

CLASS_NAMES = [
    'Airspaces', 'Communication_and_Radar', 'GPS', 'Landing_Navaids',
    'Obstacles', 'Parking', 'Procedure', 'Runway',
    'Services_and_Facilities', 'Signs_and_Lights', 'Taxiway',
    'Terminal_or_Enroute_Navaids', 'Wildlife'
]

# Final evaluation
FINAL_RUN_LABEL = "002"
FINAL_DISTILBERT_DIR = MODEL_DIR / f"distilbert_run_{FINAL_RUN_LABEL}"