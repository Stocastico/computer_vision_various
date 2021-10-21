import tensorflow_addons as tfa
from conv_mixer import *
from visualization import *

x_train, y_train, x_val, y_val, x_test, y_test = load_cifar10_dataset(0.1)

data_augmentation = specify_data_augmentation()

train_dataset = make_datasets(x_train, y_train, data_augmentation, is_train=True)
val_dataset = make_datasets(x_val, y_val, data_augmentation)
test_dataset = make_datasets(x_test, y_test, data_augmentation)


def train_model(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    hist = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )

    return hist, model, checkpoint_filepath


if __name__ == '__main__':

    # Try fixing memory allocation issues on GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPUs ???")

    # Create the model
    conv_mixer_model = get_conv_mixer_256_8()

    # Train it
    history, conv_mixer_model, checkp_file = train_model(conv_mixer_model)

    print(history.history['accuracy'])

    # save model
    conv_mixer_model.load_weights(checkp_file)
    conv_mixer_model.save('./models/conv_mixer')

    # Calc accuracy on the test set
    _, accuracy = conv_mixer_model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    # Create and save images of loss and accuracy
    plot_acc_loss_images(history, './images/model_')

    # Visualize the learned patch embeddings.
    patch_embeddings = conv_mixer_model.layers[2].get_weights()[0]
    visualization_plot(patch_embeddings, 'patch_embeddings')

    # Visualize convolution kernels
    for i, layer in enumerate(conv_mixer_model.layers):
        if isinstance(layer, layers.DepthwiseConv2D):
            if layer.get_config()["kernel_size"] == (5, 5):
                print(i, layer)

    idx = 26  # Taking a kernel from the middle of the network.

    kernel = conv_mixer_model.layers[idx].get_weights()[0]
    kernel = np.expand_dims(kernel.squeeze(), axis=2)
    visualization_plot(kernel, 'conv_kernel_' + str(idx))



