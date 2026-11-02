"""CNN without pooling. Run: python models/cnn.py"""

if __package__:
    from .cnn_common import BirdCNN, run
else:
    from cnn_common import BirdCNN, run


class CNN(BirdCNN):
    """Learn downsampling with stride-two convolutions."""

    def __init__(self, num_classes=12, input_shape=(128, 216), dropout=0.3):
        super().__init__(num_classes, input_shape, dropout, use_max_pool=False)


if __name__ == "__main__":
    run(CNN, "cnn")
