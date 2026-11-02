"""CNN with max pooling. Run: python models/cnnmax.py"""

if __package__:
    from .cnn_common import BirdCNN, run
else:
    from cnn_common import BirdCNN, run


class CNNMax(BirdCNN):
    """Downsample with max pooling after stride-one convolutions."""

    def __init__(self, num_classes=12, input_shape=(128, 216), dropout=0.3):
        super().__init__(num_classes, input_shape, dropout, use_max_pool=True)


if __name__ == "__main__":
    run(CNNMax, "cnnmax")
