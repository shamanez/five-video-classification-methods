"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None, weights=None,
          freeze_layers=False, last_trainable=-1, patience=10):
    # Set variables.
    nb_epoch = 1000
    batch_size = 16

    curtime = time.time()

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='./data/checkpoints/' + str(curtime) + '-'+ model + '.h5',
        verbose=1,
        save_best_only=True)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger('./data/logs/' + model + '-' + 'training-' + \
        str(timestamp) + '.log')

    # Helper: Early stopping.
    early_stopper = EarlyStopping(patience=patience)

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size
    val_steps_per_epoch = (len(data.data) * 0.3) // batch_size

    # Get generators.
    generator = data.frame_generator(batch_size, 'train')
    val_generator = data.frame_generator(batch_size, 'test')

    # Get the model.
    rm = ResearchModels(
        len(data.classes), model, seq_length, saved_model,
        weights=weights, freeze_layers=freeze_layers, last_trainable=last_trainable)

    # Use fit generator.
    rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[early_stopper, checkpointer, csv_logger],
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch)

def main():
    """These are the main training settings. Set each before running
    this file."""
    model = 'pretrained_lrcn'  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = 51  # int, can be 1-101 or None
    seq_length = 40
    #weights = 'data/c3d/models/sports1M_weights_tf.h5'
    weights = None
    freeze_layers = False
    last_trainable = -9

    # Chose images or features and image shape based on network.
    if model == 'conv_3d':
        data_type = 'images'
        image_shape = (112, 112, 3)
    elif model == 'lrcn' or model == 'pretrained_lrcn':
        data_type = 'image'
        image_shape = (112, 112, 3)
    else:
        image_shape = None

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape, weights=weights,
          freeze_layers=freeze_layers, last_trainable=last_trainable,
          patience=patience)

if __name__ == '__main__':
    main()
