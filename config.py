# Configuration Constants

VIDEO_FOLDER = "./data"
GROUND_TRUTH_FILE = "ground_truth.csv"
OUTPUT_FILE = "spo2_dataset.csv"

FS_MIN = 0.7  # 42 BPM
FS_MAX = 3.0  # 180 BPM
BVP_WINDOW_SEC = 180

# Configuration: Select which rPPG methods to test
RPPG_METHODS = ['POS', 'CHROM', 'ICA']

# Configuration: Select which regions to test ('ALL' or list of region names)
SELECTED_REGIONS = 'ALL'  # Can be changed to specific list like ['forehead', 'left_cheek']
