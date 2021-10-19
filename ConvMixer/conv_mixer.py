from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 128
num_epochs = 1
image_size = 32
auto = tf.data.AUTOTUNE


def load_cifar10_dataset(validation_split=0.1):
    """
    Loads the dataset used for training the network
    :return: the dataset split in train, valid and test
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    val_split = validation_split

    val_indices = int(len(x_train) * val_split)
    new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
    x_val, y_val = x_train[:val_indices], y_train[:val_indices]

    print(f"Training data samples: {len(new_x_train)}")
    print(f"Validation data samples: {len(x_val)}")
    print(f"Test data samples: {len(x_test)}")

    return new_x_train, new_y_train, x_val, y_val, x_test, y_test


def specify_data_augmentation():
    """
    Setup data augmentation steps
    :return: the data augmentation layer
    """

    data_augmentation = keras.Sequential(
        [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"), ],
        name="data_augmentation",
    )

    return data_augmentation


def make_datasets(images, labels, data_augmentation, is_train=False):
    """
    Generates the datasets
    :param images:
    :param labels:
    :param data_augmentation:
    :param is_train:
    :return:
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
        )
    return dataset.prefetch(auto)


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(image_size=32, filters=256, depth=8, kernel_size=5,
                         patch_size=2, num_classes=10):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)