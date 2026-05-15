import sys
from utils import cifar_loader, unpickle
import config

# GET LABELS AND PIXELS FROM UNPICKLE HELPER FUNC
data_batch_1 = unpickle(config.data_file_path)
meta_data = unpickle(config.meta_file)

# take the images data from batch data
images = data_batch_1["data"]
# reshape and transpose the images
images = images.reshape(len(images), 3, 32, 32).transpose(0, 2, 3, 1)
# normalize (Convert to float32 and scale to 0-1)
images = images.astype("float32") / 255.0
# take labels of the images
labels = data_batch_1["labels"]
# label names of the images
label_names = meta_data["label_names"]

input_layer = images.reshape(images.shape[0], -1)

def main() -> int:
    """Entry point for unpickling, reshaping and transposing the image data,
    ready for model injection."""
    # cifar_loader(config.images_processed_path, images)
    # cifar_loader(config.labels_processed_path, labels)
    print(input_layer.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
