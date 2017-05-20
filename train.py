"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None):
    # Set variables.
    nb_epoch = 1000000
    batch_size = 32

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='./data/checkpoints/' + model + '-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger('./data/logs/' + model + '-' + 'training-' + \
        str(timestamp) + '.log')

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
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Use fit generator.
    rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[csv_logger],
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch)

def main():
    """These are the main training settings. Set each before running
    this file."""
    model = 'conv_3d'  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = 2  # int, can be 1-101 or None
    seq_length = 40
    load_to_memory = True  # pre-load the sequences into memory

    # Chose images or features and image shape based on network.
    if model == 'conv_3d':
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model == 'lrcn':
        data_type = 'image'
        image_shape = (150, 150, 3)
    else:
        image_shape = None

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape)

if __name__ == '__main__':
    main()
