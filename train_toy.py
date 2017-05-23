"""
Train models on our toy dataset.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from toy_dataset import ToyDataset
import time

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None):
    # Set variables.
    nb_epoch = 1000000
    batch_size = 16

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

    dataset = ToyDataset(batch_size, image_shape[0], image_shape[1], 112, seq_length)

    steps_per_epoch = 10000
    val_steps_per_epoch = 1000

    # Get generators.
    generator = dataset.gen_data()

    # Get the model.
    rm = ResearchModels(3, model, seq_length, saved_model)

    # Use fit generator.
    rm.model.fit_generator(
        generator=generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[csv_logger],
        validation_data=generator,
        validation_steps=val_steps_per_epoch)

def main():
    """These are the main training settings. Set each before running
    this file."""
    model = 'conv_3d'  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = 51  # int, can be 1-101 or None
    seq_length = 16
    load_to_memory = True  # pre-load the sequences into memory

    # Chose images or features and image shape based on network.
    if model == 'conv_3d':
        data_type = 'images'
        image_shape = (112, 112, 3)
    elif model == 'lrcn':
        data_type = 'image'
        image_shape = (150, 150, 3)
    else:
        image_shape = None

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape)

if __name__ == '__main__':
    main()
