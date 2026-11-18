"""Model tests use synthetic RGB images and randomly initialized weights."""

import numpy as np
from tensorflow import keras

from cattle_breed import model


def test_efficientnet_outputs_five_probabilities():
    network = model.build_model(weights=None, input_shape=(32, 32, 3))
    images = np.random.default_rng(42).integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)

    probabilities = network(images, training=False).numpy()

    assert probabilities.shape == (20, 5)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_train_updates_backbone_only_after_head_stage():
    keras.utils.set_random_seed(42)
    network = model.build_model(weights=None, input_shape=(32, 32, 3))
    backbone = network.get_layer("backbone")
    convolution = backbone.layer.get_layer("top_conv")
    batch_norm = backbone.layer.get_layer("top_bn")
    initial_kernel = convolution.kernel.numpy().copy()
    initial_moving_mean = batch_norm.moving_mean.numpy().copy()
    images = np.random.default_rng(42).integers(0, 256, (20, 36, 36, 3), dtype=np.uint8)
    labels = np.array([0, 1, 2, 3, 4] * 3 + [1] * 5)
    head_probabilities = []

    def after_head(head_model):
        assert not backbone.trainable
        np.testing.assert_array_equal(convolution.kernel.numpy(), initial_kernel)
        np.testing.assert_allclose(head_model.optimizer.learning_rate.numpy(), 1e-3)
        assert head_model.loss.label_smoothing == 0.1
        head_probabilities.append(head_model(images[:5, 2:-2, 2:-2], training=False).numpy())

    epochs = model.train(
        network,
        (images[5:], labels[5:]),
        (images[:5], labels[:5]),
        head_epochs=1,
        ft_epochs=1,
        on_head_trained=after_head,
        verbose=0,
    )

    assert epochs == {"head_epochs": 1, "ft_epochs": 1}
    assert len(head_probabilities) == 1
    assert backbone.trainable
    assert not np.array_equal(convolution.kernel.numpy(), initial_kernel)
    np.testing.assert_array_equal(batch_norm.moving_mean.numpy(), initial_moving_mean)
    np.testing.assert_allclose(network.optimizer.learning_rate.numpy(), 1e-5)
    assert network.loss.label_smoothing == 0.1


def test_train_balances_present_classes_without_validation():
    inputs = keras.Input((32, 32, 3))
    backbone = keras.Sequential([keras.layers.GlobalAveragePooling2D()], name="backbone")
    probabilities = keras.layers.Dense(
        5, activation="softmax", bias_initializer=keras.initializers.Constant([2, 0, 0, 0, 0])
    )(backbone(inputs))
    network = keras.Model(inputs, probabilities)
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    first_epoch_losses = []

    def after_head(head_model):
        first_epoch_losses.extend(head_model.history.history["loss"])

    epochs = model.train(
        network, (images, np.array([0, 0, 0, 4])),
        head_epochs=2, ft_epochs=1, on_head_trained=after_head, verbose=0,
    )

    # Balanced weighting gives the two observed classes equal influence.
    # Their mean smoothed target assigns 0.47 to class 0, the sole nonzero logit.
    expected_loss = np.log(np.exp(2) + 4) - 2 * 0.47
    np.testing.assert_allclose(first_epoch_losses[0], expected_loss, atol=1e-6)
    assert epochs == {"head_epochs": 2, "ft_epochs": 1}


def test_gradcam_for_blank_image_after_serialization(tmp_path):
    network = model.build_model(weights=None)
    path = tmp_path / "network.keras"
    network.save(path)
    network = keras.models.load_model(path, compile=False)

    heatmap = model.gradcam(network, np.zeros((224, 224, 3), dtype=np.uint8), class_idx=2)

    assert heatmap.shape == (224, 224)
    assert np.all(np.isfinite(heatmap))
    assert np.all((heatmap >= 0) & (heatmap <= 1))
