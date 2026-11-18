"""EfficientNet-B0 classifier, two-stage training, and Grad-CAM for RGB images."""

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from . import config


@keras.utils.register_keras_serializable(package="cattle_breed")
class InferenceBackbone(keras.layers.Wrapper):
    """Apply the backbone with fixed BatchNorm statistics in both stages."""

    def call(self, inputs):
        # A Functional model can propagate its training flag to nested models.
        # This wrapper consumes only inputs and explicitly sets the inner mode.
        return self.layer(inputs, training=False)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)


def build_model(weights="imagenet", input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)):
    """Build a classifier that accepts RGB values in 0–255 and outputs class probabilities."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.SEED)
    inputs = keras.Input(shape=input_shape, name="image")
    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(config.AUG_ROTATION),
            keras.layers.RandomBrightness(config.AUG_BRIGHTNESS, value_range=(0, 255)),
            keras.layers.RandomContrast(config.AUG_CONTRAST),
        ],
        name="augmentation",
    )
    efficientnet = keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=input_shape
    )
    # EfficientNet handles rescaling internally.
    backbone = InferenceBackbone(efficientnet, name="backbone")
    backbone.trainable = False
    features = backbone(augmentation(inputs))
    pooled = keras.layers.GlobalAveragePooling2D(name="feature_pool")(features)
    dropped = keras.layers.Dropout(config.DROPOUT)(pooled)
    logits = keras.layers.Dense(len(config.CLASSES), name="logits")(dropped)
    probabilities = keras.layers.Activation("softmax", dtype="float32", name="probabilities")(logits)
    return keras.Model(inputs, probabilities, name="cattle_breed_classifier")


def _batches(images, labels, input_shape, *, training):
    height, width, channels = input_shape

    def prepare(image, label):
        if training:
            image = tf.image.random_crop(image, input_shape)
        else:
            image = tf.image.resize_with_crop_or_pad(image, height, width)
        image = tf.ensure_shape(image, (height, width, channels))
        return image, tf.one_hot(label, len(config.CLASSES))

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(len(labels), seed=config.SEED)
    return dataset.map(prepare).batch(config.BATCH_SIZE).prefetch(1)


def _fit_stage(model, batches, val_batches, class_weights, lr, epochs, patience, verbose):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    stopping = None
    if val_batches is not None:
        stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=patience, restore_best_weights=True
        )
    history = model.fit(
        batches,
        validation_data=val_batches,
        class_weight=class_weights,
        epochs=epochs,
        shuffle=False,
        callbacks=[stopping] if stopping is not None else [],
        verbose=verbose,
    )
    return stopping.best_epoch + 1 if stopping is not None else len(history.epoch)


def train(
    model,
    train,
    val=None,
    *,
    head_epochs=config.HEAD_EPOCHS,
    ft_epochs=config.FT_EPOCHS,
    on_head_trained=None,
    verbose=1,
):
    """Train the head (E1) and then fine-tune the whole model (E2).

    Supply each split as (RGB images, integer labels). Images are randomly cropped
    for training and center-cropped for validation. The callback receives the
    model after E1 and before E2. Return the epoch selected in each stage, or the
    number of epochs run if validation was omitted.
    """
    if head_epochs < 1 or ft_epochs < 1:
        raise ValueError("Both training stages require at least one epoch")
    train_images, labels = train
    labels = np.asarray(labels)
    if (
        labels.ndim != 1
        or not np.issubdtype(labels.dtype, np.integer)
        or not labels.size
        or labels.min() < 0
        or labels.max() >= len(config.CLASSES)
    ):
        raise ValueError("Training labels must be a nonempty vector of class indices")
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weights = {int(index): float(weight) for index, weight in zip(classes, weights)}
    input_shape = model.input_shape[1:]
    batches = _batches(train_images, labels, input_shape, training=True)
    val_batches = _batches(*val, input_shape, training=False) if val is not None else None

    backbone = model.get_layer("backbone")
    backbone.trainable = False
    best_head_epoch = _fit_stage(
        model, batches, val_batches, class_weights,
        config.HEAD_LR, head_epochs, config.HEAD_PATIENCE, verbose,
    )
    if on_head_trained is not None:
        on_head_trained(model)

    backbone.trainable = True
    best_ft_epoch = _fit_stage(
        model, batches, val_batches, class_weights,
        config.FT_LR, ft_epochs, config.FT_PATIENCE, verbose,
    )
    return {"head_epochs": best_head_epoch, "ft_epochs": best_ft_epoch}


def gradcam(model, image, class_idx):
    """Return a class-specific heatmap in [0, 1] for one RGB image at model input size."""
    if not 0 <= class_idx < len(config.CLASSES):
        raise ValueError("Class index is outside the classifier's output range")
    # The pooling layer consumes the backbone's last spatial activations. Use
    # logits for the class score to avoid gradients saturating after softmax.
    feature_model = keras.Model(
        model.inputs, [model.get_layer("feature_pool").input, model.get_layer("logits").output]
    )
    image_batch = tf.cast(image[None, ...], tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image_batch)
        features, logits = feature_model(image_batch, training=False)
        score = logits[:, class_idx]
    gradients = tape.gradient(score, features)
    channel_weights = tf.reduce_mean(tf.cast(gradients, tf.float32), axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(tf.cast(features, tf.float32) * channel_weights, axis=-1))
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    heatmap = tf.image.resize(heatmap[..., None], image_batch.shape[1:3])
    return heatmap[0, ..., 0].numpy()
