"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os.path
from data import DataSet
from extractor import Extractor
from tqdm import tqdm

# Set defaults.
seq_length = 40
class_limit = 2  # Number of classes to extract. Can be 1-101 or None for all.

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit)

# get the model.
model = Extractor()

# Loop through data.
pbar = tqdm(total=len(data.data))
for video in data.data:

    # Get the path to the sequence for this video.
    path = './data/MYsequences/' + video[2] + '-' + str(seq_length) + \
        '-features.txt'

    # Check if we already have it.
    if os.path.isfile(path):
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)

    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)   #set each example is for 40 frames . Fixed length seqyebce 
  

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:   #take each image in the 
        
        features = model.extract(image) #send the each image to extract 
        sequence.append(features)
        np.savetxt(path, sequence)
    # Save the sequence.
    print(len(sequence))
     
    np.savetxt(path, sequence)

    pbar.update(1)
   
pbar.close()
