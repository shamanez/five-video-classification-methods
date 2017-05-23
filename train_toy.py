"""
Train models on our toy dataset.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
from toy_dataset import ToyDataset
import time

def train(seq_length, model, saved_model=None,
          class_limit=None, image_shape=None):
    # Set variables.
    nb_epoch = 1000
    batch_size = 16

    curtime = time.time()

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='./data/checkpoints/' + str(curtime) + '-' + model + '.h5',
        verbose=1,
        save_best_only=True)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger('./data/logs/' + model + '-' + 'training-' + \
        str(timestamp) + '.log')

    dataset = ToyDataset(batch_size, image_shape[0], image_shape[1], seq_length)

    steps_per_epoch = 500
    val_steps_per_epoch = 50

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
    model = 'pretrained_lrcn'  # see `models.py` for more
    saved_model = None  # None or weights file
    class_limit = 51  # int, can be 1-101 or None
    seq_length = 16
    load_to_memory = True  # pre-load the sequences into memory
    image_shape = (112, 112, 3)

    train(seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape)

if __name__ == '__main__':
    main()
