from typing import Final
from pathlib import Path

# FILE PATHS
data_dir: Final[Path] = Path(__file__).parent.parent / "data" / "cifar-10-batches-py"
data_file_path: Final[Path] = data_dir / "data_batch_1"
meta_file: Final[Path] = data_dir / "batches.meta"

images_processed_path: Final[Path] = Path(__file__).parent.parent / "data" / "cifar_images_processed.npy"
labels_processed_path: Final[Path] = Path(__file__).parent.parent / "data" / "cifar_labels_processed.npy"

# NEURONS
DL_512_COUNT = 512 # Dense layer of 512 neurons

# LAYERS
LAYER_DIMS = [3072, 512, 512, 512, 10]
