from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models


IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3


# ============================================================
# DATA LOADING
# ============================================================

def load_image_mask(image_path, mask_path):
    # MRI image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(
        image,
        [IMG_HEIGHT, IMG_WIDTH],
        method="bilinear"
    )
    image = tf.cast(image, tf.float32) / 255.0

    # Ground-truth segmentation mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(
        mask,
        [IMG_HEIGHT, IMG_WIDTH],
        method="nearest"
    )
    mask = tf.cast(mask, tf.float32) / 255.0

    # Force binary mask
    mask = tf.cast(mask > 0.5, tf.float32)

    return image, mask


def make_dataset(
    image_dir,
    mask_dir,
    batch_size=8,
    shuffle=False
):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_paths = sorted(image_dir.glob("*.jpg"))

    if not image_paths:
        raise RuntimeError(
            f"No JPG images found in: {image_dir}"
        )

    mask_paths = []

    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for {image_path.name}"
            )

        mask_paths.append(mask_path)

    image_paths = [str(p) for p in image_paths]
    mask_paths = [str(p) for p in mask_paths]

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, mask_paths)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=42,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image_mask,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# SEGMENTATION METRICS
# ============================================================

@tf.keras.utils.register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    return (
        2.0 * intersection + 1e-6
    ) / (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        + 1e-6
    )


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(
        y_true,
        y_pred
    )


@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    bce = tf.reduce_mean(bce)

    return bce + dice_loss(
        y_true,
        y_pred
    )


@tf.keras.utils.register_keras_serializable()
def iou_score(y_true, y_pred):
    y_true = tf.cast(
        y_true > 0.5,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred > 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    union = (
        tf.reduce_sum(y_true)
        + tf.reduce_sum(y_pred)
        - intersection
    )

    return (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )


# ============================================================
# U-NET
# ============================================================

def conv_block(inputs, filters):
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    return x


def build_unet():
    inputs = layers.Input(
        shape=(
            IMG_HEIGHT,
            IMG_WIDTH,
            IMG_CHANNELS
        )
    )

    # Encoder
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    # Bottleneck
    bn = conv_block(p4, 512)

    # Decoder
    u4 = layers.Conv2DTranspose(
        256,
        2,
        strides=2,
        padding="same"
    )(bn)

    u4 = layers.Concatenate()(
        [u4, c4]
    )

    c5 = conv_block(u4, 256)

    u3 = layers.Conv2DTranspose(
        128,
        2,
        strides=2,
        padding="same"
    )(c5)

    u3 = layers.Concatenate()(
        [u3, c3]
    )

    c6 = conv_block(u3, 128)

    u2 = layers.Conv2DTranspose(
        64,
        2,
        strides=2,
        padding="same"
    )(c6)

    u2 = layers.Concatenate()(
        [u2, c2]
    )

    c7 = conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(
        32,
        2,
        strides=2,
        padding="same"
    )(c7)

    u1 = layers.Concatenate()(
        [u1, c1]
    )

    c8 = conv_block(u1, 32)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c8)

    return models.Model(
        inputs,
        outputs,
        name="NeuroVision_UNet"
    )
