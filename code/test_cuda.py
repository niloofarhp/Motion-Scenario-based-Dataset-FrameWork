import tensorflow as tf
import torch


def test_tensorflow():
    print("TensorFlow version:", tf.__version__)
    if tf.test.gpu_device_name():
        print("TensorFlow GPU device found:", tf.test.gpu_device_name())
    else:
        print("No TensorFlow GPU device found.")


def test_torch():
    print("Torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("Torch GPU device found:", torch.cuda.get_device_name(0))
    else:
        print("No Torch GPU device found.")


if __name__ == "__main__":
    test_tensorflow()
    test_torch()
