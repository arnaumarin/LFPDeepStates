import numpy as np
concatenated_data = np.load("data_Phoebe_210626.npy")
concatenated_data = concatenated_data.transpose()
print("Shape of the training data:", concatenated_data.shape)

import pickle
path = 'metadata_Phoebe_210626.pkl'
with open(path, 'rb') as f:
    metadata=pickle.load(f)
    labels = metadata['labels']
    time_indices = metadata['time_indices']
    probs = metadata['prob']
    # sort the indices

concatenated_states = labels
print("Shape of the concatenated states:", concatenated_states.shape)

# Path: AE/ae.py
