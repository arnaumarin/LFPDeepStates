import glob
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.optim as optim
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Reshape
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
import keras.callbacks
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam

def normalize_data(data):
    return (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)

def labels_to_numbers(labels):
    mapping = {
        'awake': 0,
        'slow_updown': 1,
        'asynch_MA': 2,
        'slow_MA': 2, 
    }
    return [mapping[label] for label in labels]

def labels_to_numbersMA(labels):
    mapping = {
        'awake': 0,
        'slow_updown': 1,
        'asynch_MA': 2,
        'slow_MA': 3, 
        'unknown': 4, 
    }
    return [mapping[label] for label in labels]

def read_data_and_metadata(NAME, TIME, best_channel=True):
    data_file = f"{NAME}_{TIME}_fullrec.pkl"

    try:
        data = pd.read_pickle(data_file)
        # define number of samples
        num_samples = 2000
        # define initial matrix
        mat = np.array([])
        # define state label list
        state_label = []
        # get unique states
        states = [i for i in np.unique(data['states']) if i != 4]
        # get indices of unique states
        tdx = {}
        for state in states: 
            tdx[state] = np.where(data['states']==state)

        if best_channel == True:
            cdx = data['labels'].index(data['best_ch'])
            
            # loop over states
            for state in states:
                trace = data['value'][cdx, :][tdx[state]]
                # get minimum number of timeseries
                num_ts = int(np.floor(len(trace)/num_samples))
                # reshape
                mat_add = trace[:num_ts * num_samples].reshape((num_ts, num_samples))
                # stack to original data matrix
                mat = np.vstack([mat, mat_add]) if mat.size else mat_add
                # add state_label
                state_label.extend([data['state_correspondence'][state]] * mat_add.shape[0])
        
        elif best_channel==False:
            for cdx in range(data['value'].shape[0]):
                # loop over states
                for state in states:
                    trace = data['value'][cdx, :][tdx[state]]
                    # get minimum number of timeseries
                    num_ts = int(np.floor(len(trace)/num_samples))
                    # reshape
                    mat_add = trace[:num_ts * num_samples].reshape((num_ts, num_samples))
                    # stack to original data matrix
                    mat = np.vstack([mat, mat_add]) if mat.size else mat_add
                    # add state_label
                    state_label.extend([data['state_correspondence'][state]] * mat_add.shape[0])
        else:
            cdx = data['labels'].index(best_channel)
            
            # loop over states
            for state in states:
                trace = data['value'][cdx, :][tdx[state]]
                # get minimum number of timeseries
                num_ts = int(np.floor(len(trace)/num_samples))
                # reshape
                mat_add = trace[:num_ts * num_samples].reshape((num_ts, num_samples))
                # stack to original data matrix
                mat = np.vstack([mat, mat_add]) if mat.size else mat_add
                # add state_label
                state_label.extend([data['state_correspondence'][state]] * mat_add.shape[0])

        return mat, state_label

    except FileNotFoundError:
        print(f"Files for NAME={NAME} and TIME={TIME} not found.")
        return None, None

def read_data_and_metadata_and_time(NAME, TIME):
    data_file = f"{NAME}_{TIME}_fullrec.pkl"

    try:
        data = pd.read_pickle(data_file)
        # define number of samples
        num_samples = 2000
        # define initial matrix
        mat = np.array([])
        # define state label list
        state_label = []
        # initalize temporal index
        time_indices = np.array([])
        # get unique states
        states = [i for i in np.unique(data['states']) if i != 4]
        # get indices of unique states
        tdx = {}
        for state in states: 
            tdx[state] = np.where(data['states']==state)

        cdx = data['labels'].index(data['best_ch'])
        
        # loop over states
        for state in states:
            trace = data['value'][cdx, :][tdx[state]]
            # get minimum number of timeseries
            num_ts = int(np.floor(len(trace)/num_samples))
            # reshape
            mat_add = trace[:num_ts * num_samples].reshape((num_ts, num_samples))
            # stack to original data matrix
            mat = np.vstack([mat, mat_add]) if mat.size else mat_add
            # add state_label
            state_label.extend([data['state_correspondence'][state]] * mat_add.shape[0])
            # check if temporal index
            idx_state = tdx[state][0].copy()
            idx_state = idx_state[:num_ts * num_samples].reshape((num_ts, num_samples))
            idx_state = idx_state[:, 0]
            time_indices = np.concatenate([time_indices, idx_state])
    
       

        return mat, state_label, time_indices

    except FileNotFoundError:
        print(f"Files for NAME={NAME} and TIME={TIME} not found.")
        return None, None

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Second convolutional block
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Third convolutional block
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 250, 256),  # Adjusted based on pooling and data dimensions
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def plot_loss_accuracy(losses, accuracies):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(losses)
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss over epochs")

    ax[1].plot(accuracies)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_title("Accuracy over epochs")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    disp.plot()
    plt.show()

def sinusoidal_activation(x):
    return tf.sin(x)

def plot_training_history(history):
    plt.figure(figsize=(8, 5))
    # Plotting accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_model_1(name_testing_subject, data_normalized, labels_numeric):
    # Step 1: Normalize Data
    data_normalized = data_normalized

    # Step 2: Convert labels
    labels_numeric = labels_numeric
    # labels_numeric_MA = labels_to_numbersMA(training_states)  # Note: This isn't used in the snippet

    # Step 3: Create Tensor Datasets
    dataset = TensorDataset(
        torch.tensor(data_normalized, dtype=torch.float32).unsqueeze(1),
        torch.tensor(labels_numeric, dtype=torch.long)
    )

    # Step 4: Set Up DataLoader
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Step 5: Model, Loss, and Optimizer Initialization
    device = torch.device("cpu")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 6: Training Loop
    num_epochs = 6
    losses, accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / (batch_idx + 1)
        epoch_acc = 100. * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.2f}%")

        # Save models every epoch
        if (epoch + 1) % 2 == 0:
            primary_model_save_name = f"models_trainmain/primary_model_epoch_{name_testing_subject}_{epoch + 1}.pth"
            torch.save(model.state_dict(), primary_model_save_name)
            print(f"Saved primary model at epoch {epoch + 1} as {primary_model_save_name}")

    print("Training of primary model complete.")
    return model  # Optionally, return the trained model for use outside the function

def model1(name_testing_subject, training_data, training_states, testing_data, testing_states, treshold_mod1, norm=False):
    if norm:
        test_data_normalized = normalize_data(testing_data)
        train_data_normalized = normalize_data(training_data)
    else:
        test_data_normalized = testing_data
        train_data_normalized = training_data

    train_labels_numeric = labels_to_numbers(training_states)  # no difference between MA_states

    test_labels_numeric = labels_to_numbers(testing_states)
    test_dataset = TensorDataset(torch.tensor(test_data_normalized, dtype=torch.float32).unsqueeze(1),
                                 torch.tensor(test_labels_numeric, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CNN()
    num_ech = 3

    model_path = f"models_trainmain/primary_model_epoch_{name_testing_subject}_{num_ech}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        model = train_model_1(name_testing_subject, train_data_normalized, train_labels_numeric)
        print(f"Trained a new model as no saved model was found for epoch {num_ech}")

    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds_prob, all_labels_prob = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            probabilities = F.softmax(outputs, dim=1)
            _, preds = torch.max(probabilities, 1)
            preds[probabilities.max(dim=1).values < treshold_mod1] = -1
            all_preds_prob.extend(preds.cpu().numpy())
            all_labels_prob.extend(labels.numpy())
        

    cm = confusion_matrix(all_labels_prob, all_preds_prob, labels=[0, 1, 2, -1])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(7, 4))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=["awake", "slow_updown", "MA states", 'Unknown'],
                yticklabels=["awake", "slow_updown", "MA states", 'Unknown'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    unknown_counts = {i: np.sum((np.array(all_labels_prob) == i) & (np.array(all_preds_prob) == -1)) for i in [0, 1, 2]}

    # plt.figure(figsize=(4, 4))
    # plt.bar(unknown_counts.keys(), unknown_counts.values(), tick_label=["awake", "slow_updown", "MA states"])
    # plt.xlabel('Original Class')
    # plt.ylabel('Count of Unknown Predictions')
    # plt.title('Histogram of Unknown Predictions per Class')
    # plt.show()

    labels_train_2 = [labels_numeric_MA[i] for i in indices_train_2]
    labels_train_3 = [labels_numeric_MA[i] for i in indices_train_3]

    # Step 4: Filter the testing data based on the indices
    data_normalized_test_2 = data_normalized_test[indices_test_2]
    data_normalized_test_3 = data_normalized_test[indices_test_3]
    labels_test_2 = [labels_numeric_MA_test[i] for i in indices_test_2]
    labels_test_3 = [labels_numeric_MA_test[i] for i in indices_test_3]

    # print("Filtered data based on labels 2 and 3.")
    # Step 1: Combine the training data for states 2 an
    return cm, unknown_counts


def mod2(name_testing_subject, training_data, training_states, testing_data, testing_states, treshold_mod2, LEAVEONEOUT, norm=False):
    name = name_testing_subject
    
    if norm == True:
        data_normalized = normalize_data(training_data)
        data_normalized_test = normalize_data(testing_data)
    else:
        data_normalized = training_data
        data_normalized_test = testing_data

    labels_numeric_MA = labels_to_numbersMA(training_states)
    labels_numeric_MA_test = labels_to_numbersMA(testing_states)

    # Step 1: Find the indices in the training labels where the state == 2 or 3
    indices_train_2 = np.where(np.array(labels_numeric_MA) == 2)[0].tolist()
    indices_train_3 = np.where(np.array(labels_numeric_MA) == 3)[0].tolist()

    # Step 2: Find the indices in the testing labels where the state == 2 or 3
    indices_test_2 = np.where(np.array(labels_numeric_MA_test) == 2)[0].tolist()
    indices_test_3 = np.where(np.array(labels_numeric_MA_test) == 3)[0].tolist()

    # Step 3: Filter the training data based on the indices
    data_normalized_train_2 = data_normalized[indices_train_2]
    data_normalized_train_3 = data_normalized[indices_train_3]
    data_normalized_train_combined = np.concatenate((data_normalized_train_2, data_normalized_train_3), axis=0)
    labels_train_combined = labels_train_2 + labels_train_3

    # Step 2: Combine the testing data for states 2 and 3
    data_normalized_test_combined = np.concatenate((data_normalized_test_2, data_normalized_test_3), axis=0)
    labels_test_combined = labels_test_2 + labels_test_3

    # Convert the lists to numpy arrays
    labels_train_combined = np.array(labels_train_combined)
    labels_test_combined = np.array(labels_test_combined)

    # Adjust the labels
    labels_train_combined = (labels_train_combined - 2)  # Convert 2->0 and 3->1
    labels_test_combined = (labels_test_combined - 2)  # Convert 2->0 and 3->1

    # Step 3: Print the shapes of the combined datasets and labels
    # print("Shape of combined training data:", data_normalized_train_combined.shape)
    # print("Length of combined training labels:", len(labels_train_combined))
    # print("Shape of combined testing data:", data_normalized_test_combined.shape)
    # print("Length of combined testing labels:", len(labels_test_combined))

    data_normalized_train_combined_reshaped = np.reshape(data_normalized_train_combined, (-1, 2000, 1))
    data_normalized_test_combined_reshaped = np.reshape(data_normalized_test_combined, (-1, 2000, 1))

    # Splitting the training dataset into training and validation sets using an 80-20 split.
    data_normalized_train, data_normalized_val, labels_train, labels_val = train_test_split(
        data_normalized_train_combined_reshaped,
        labels_train_combined, test_size=0.2,
        random_state=42)
    # print(f"Training data shape: {data_normalized_train.shape}")
    # print(f"Validation data shape: {data_normalized_val.shape}")

    from keras.models import load_model
    epch = 10
    # Assuming "name" and "epoch" variables are defined elsewhere in your code.
    if not LEAVEONEOUT:
        model_path = f"models_MAstates/model_MAstates_{name}_{epch}.h5"
    else:
        model_path = f"models_MAstates/model_MAstates_{name}_transflearning7subs_{epch}.h5"

    # Check if model file exists and load it; else, define and train the model
    if os.path.exists(model_path):
        model = load_model(model_path, custom_objects={'sinusoidal_activation': sinusoidal_activation})
        print(f"Loaded model from {model_path}")
    else:
        model = Sequential([
            Reshape((2000, 1), input_shape=(2000,)),
            Conv1D(50, kernel_size=3, activation=sinusoidal_activation),
            Conv1D(32, kernel_size=5, activation=sinusoidal_activation),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(model.summary())

        # Custom callback to save the model every 5 epochs
        class CustomSaver(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if epoch % 5 == 4:  # Save model at end of every 5th epoch (in this case, epochs 5 and  10)
                    self.model.save(f"models_MAstates/model_MAstates_{name}_{epoch + 1}.h5")
                    print(f"Saved model_MAstates_{name}_{epoch + 1}.h5 at epoch {epoch + 1}")

        history = model.fit(data_normalized_train, labels_train,
                            validation_data=(data_normalized_val, labels_val),
                            epochs=10, batch_size=32, callbacks=[CustomSaver()])

        plot_training_history(history)

    # loss, accuracy = model.evaluate(data_normalized_test_combined_reshaped, labels_test_combined)
    # print(f"Test Loss: {loss:.2f}, Test Accuracy: {accuracy * 100:.2f}%")
    y_pred = (model.predict(data_normalized_test_combined_reshaped) > 0.5).astype("int32")
    # cm = confusion_matrix(labels_test_combined, y_pred)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # accuracies = np.diag(cm_normalized)
    # classes = ['Asynch_MA', 'Slow_MA']

    # Plot the confusion matrix without unknowns
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title(f'{name} classification of MA states')
    # plt.show()

    # Get the softmax outputs from the model
    predictions = model.predict(data_normalized_test_combined_reshaped)
    predicted_classes = np.where(predictions > treshold_mod2, 1,
                                np.where(predictions <= 1 - treshold_mod2, 0, -1)).flatten()
    true_classes = labels_test_combined

    return y_pred, predictions, true_classes, data_normalized_test_combined, labels_test_combined

def mod2_onlypredict(name_testing_subject, testing_data, testing_states, treshold_mod2, norm=False):
    name = name_testing_subject
    
    if norm == True:
        data_normalized_test = normalize_data(testing_data)
    else:
        data_normalized_test = testing_data

    labels_numeric_MA_test = labels_to_numbersMA(testing_states)

    # Step 2: Find the indices in the testing labels where the state == 2 or 3
    indices_test_2 = np.where(np.array(labels_numeric_MA_test) == 2)[0].tolist()
    indices_test_3 = np.where(np.array(labels_numeric_MA_test) == 3)[0].tolist()


    # Step 4: Filter the testing data based on the indices
    data_normalized_test_2 = data_normalized_test[indices_test_2]
    data_normalized_test_3 = data_normalized_test[indices_test_3]
    labels_test_2 = [labels_numeric_MA_test[i] for i in indices_test_2]
    labels_test_3 = [labels_numeric_MA_test[i] for i in indices_test_3]

    # Step 2: Combine the testing data for states 2 and 3
    data_normalized_test_combined = np.concatenate((data_normalized_test_2, data_normalized_test_3), axis=0)
    labels_test_combined = labels_test_2 + labels_test_3

    # Convert the lists to numpy arrays
    labels_test_combined = np.array(labels_test_combined)

    # Adjust the labels
    labels_test_combined = (labels_test_combined - 2)  # Convert 2->0 and 3->1

    data_normalized_test_combined_reshaped = np.reshape(data_normalized_test_combined, (-1, 2000, 1))

    from keras.models import load_model
    epch = 10
    # Assuming "name" and "epoch" variables are defined elsewhere in your code.
    model_path = f"models_MAstates/model_MAstates_{name}_{epch}.h5"

    # Check if model file exists and load it; else, define and train the model
    model = load_model(model_path, custom_objects={'sinusoidal_activation': sinusoidal_activation})
    print(f"Loaded model from {model_path}")


    # loss, accuracy = model.evaluate(data_normalized_test_combined_reshaped, labels_test_combined)
    # print(f"Test Loss: {loss:.2f}, Test Accuracy: {accuracy * 100:.2f}%")
    y_pred = (model.predict(data_normalized_test_combined_reshaped) > 0.5).astype("int32")
    
    # Get the softmax outputs from the model
    predictions = model.predict(data_normalized_test_combined_reshaped)
    predicted_classes = np.where(predictions > treshold_mod2, 1,
                                np.where(predictions <= 1 - treshold_mod2, 0, -1)).flatten()
    true_classes = labels_test_combined

    return y_pred, predictions, true_classes, data_normalized_test_combined, labels_test_combined

def mod2_onlypredict_and_time(name_testing_subject, testing_data, testing_states, treshold_mod2, testing_indices, norm=False):
    name = name_testing_subject
    
    if norm == True:
        data_normalized_test = normalize_data(testing_data)
    else:
        data_normalized_test = testing_data

    labels_numeric_MA_test = labels_to_numbersMA(testing_states)

    # Step 2: Find the indices in the testing labels where the state == 2 or 3
    indices_test_2 = np.where(np.array(labels_numeric_MA_test) == 2)[0].tolist()
    indices_test_3 = np.where(np.array(labels_numeric_MA_test) == 3)[0].tolist()

    # Step 4: Filter the testing data based on the indices
    data_normalized_test_2 = data_normalized_test[indices_test_2]
    data_normalized_test_3 = data_normalized_test[indices_test_3]
    labels_test_2 = [labels_numeric_MA_test[i] for i in indices_test_2]
    labels_test_3 = [labels_numeric_MA_test[i] for i in indices_test_3]
    indices_test_2 = [testing_indices[i] for i in indices_test_2]
    indices_test_3 = [testing_indices[i] for i in indices_test_3]

    # Step 2: Combine the testing data for states 2 and 3
    data_normalized_test_combined = np.concatenate((data_normalized_test_2, data_normalized_test_3), axis=0)
    labels_test_combined = labels_test_2 + labels_test_3
    indices_test_combined = np.array(indices_test_2 + indices_test_3)

    # Convert the lists to numpy arrays
    labels_test_combined = np.array(labels_test_combined)

    # Adjust the labels
    labels_test_combined = (labels_test_combined - 2)  # Convert 2->0 and 3->1

    data_normalized_test_combined_reshaped = np.reshape(data_normalized_test_combined, (-1, 2000, 1))

    from keras.models import load_model
    epch = 10
    # Assuming "name" and "epoch" variables are defined elsewhere in your code.
    model_path = f"models_MAstates/model_MAstates_{name}_{epch}.h5"

    # Check if model file exists and load it; else, define and train the model
    model = load_model(model_path, custom_objects={'sinusoidal_activation': sinusoidal_activation})
    print(f"Loaded model from {model_path}")

    # loss, accuracy = model.evaluate(data_normalized_test_combined_reshaped, labels_test_combined)
    # print(f"Test Loss: {loss:.2f}, Test Accuracy: {accuracy * 100:.2f}%")
    y_pred = (model.predict(data_normalized_test_combined_reshaped) > 0.5).astype("int32")
    
    # Get the softmax outputs from the model
    predictions = model.predict(data_normalized_test_combined_reshaped)
    predicted_classes = np.where(predictions > treshold_mod2, 1,
                                np.where(predictions <= 1 - treshold_mod2, 0, -1)).flatten()
    true_classes = labels_test_combined

    return y_pred, predictions, true_classes, data_normalized_test_combined, labels_test_combined, indices_test_combined

def organize_datamodel2(testing_data, testing_states, df_model2_sub, indicesMA):
    
    data_normalized_test = normalize_data(testing_data)
    labels_numeric_MA_test = labels_to_numbersMA(testing_states)

    # Step 2: Combine the testing data for states 2 and 3
    data_normalized_test = data_normalized_test[indicesMA, :]

    # Convert the lists to numpy arrays
    labels_test_combined = np.array([labels_numeric_MA_test[i] for i in indicesMA])
    
    # get correct order of indices
    time_indices = [df_model2_sub['time_index'].iloc[i] for i in range(data_normalized_test.shape[0])]

    return data_normalized_test, labels_test_combined, time_indices

def get_nametimelist(matching_files):
    names_dict = {}
    times_dict = {}

    for file_path in matching_files:
        file_name = file_path.split("/")[-1]
        NAME = file_path.split('_')[0]
        TIME = file_path.split('_')[1]  # Remove the file extension
        if NAME not in names_dict:
            names_dict[NAME] = []
        if NAME not in times_dict:
            times_dict[NAME] = []
        names_dict[NAME].append(TIME)
        times_dict[NAME].append(TIME)
                
    name_time_list = []
    for name, times in times_dict.items():
        for time in times:
            name_time_list.append((name, time))

    return name_time_list

def train_autoencoder(data, encoding_dim=55, epochs=250, batch_size=64):



    """
    Train an autoencoder for one-dimensional data.
    Arguments:
    data -- Training data, a 2D array-like where each row is a sample.
    encoding_dim -- Number of neurons in the bottleneck layer (default 55).
    epochs -- Number of epochs to train for (default 50).
    batch_size -- Number of samples per batch (default 256).
    Returns:
    autoencoder -- A trained model.
    """
    # Define the autoencoder structure
    autoencoder = Sequential()
    autoencoder.add(Dense(encoding_dim, activation='relu', input_shape=(data.shape[1],)))
    autoencoder.add(Dense(data.shape[1], activation='linear'))
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True)

    return autoencoder

def train_autoencoder_validation(data, validation_data, encoding_dim=55, epochs=250, batch_size=64):
    """
    Train an autoencoder for one-dimensional data.
    ... [rest of the docstring]
    Returns:
    autoencoder -- A trained model.
    history -- Training history containing the loss over epochs.
    """
    # Define the autoencoder structure
    autoencoder = Sequential()
    autoencoder.add(Dense(encoding_dim, activation='relu', input_shape=(data.shape[1],)))
    autoencoder.add(Dense(data.shape[1], activation='linear'))
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    history = autoencoder.fit(data, data, validation_data=(validation_data, validation_data), epochs=epochs, batch_size=batch_size, shuffle=True)
    return autoencoder, history