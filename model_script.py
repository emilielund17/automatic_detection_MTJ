import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def load_data_coord(image_path, label_path):
    X = np.load(image_path)
    Y = np.load(label_path)

    X = X[..., np.newaxis]  # Shape (num_samples, height, width, 1) 

    Y = Y.astype(np.float32)

    img_height, img_width = X.shape[1], X.shape[2]
    
    # Normalize ground truth coordinates
    Y[:, 0] /= img_width   # Normalize x (center)
    Y[:, 1] /= img_height  # Normalize y (center)

    return X, Y

def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):

    shortcut = x
    if x.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding="same", kernel_regularizer=l2(0.001))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, kernel_size, strides=strides, padding="same", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (5, 5), strides=(1, 1), padding="same", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    x = Add()([shortcut, x])
    x = Activation("relu")(x)

    return x

def build_model(input_shape, filter_size, stride_size):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (filter_size, filter_size), strides=stride_size, padding="same", activation="relu", kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (filter_size, filter_size), strides=stride_size, padding="same", activation="relu", kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # x = residual_block(x, filters=16)
    # x = MaxPooling2D((2, 2))(x)

    # x = residual_block(x, filters=16)
    # x = MaxPooling2D((2, 2))(x)

    # x = residual_block(x, filters=32)

    # x = residual_block(x, filters=32)

    x = Flatten()(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(2, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    return model

def find_label_paths(data_paths, label_dir):
    label_paths = []
    for data_path in data_paths:
        # Extract the xyz part of the filename
        file_name = os.path.basename(data_path).replace("_processed.npy", "").replace("_filtered.npy", "")
        
        # Construct the corresponding label path
        label_path = os.path.join(label_dir, f"{file_name}_trk.npy")
        
        # Check if the label file exists
        if os.path.exists(label_path):
            label_paths.append(label_path)
        else:
            print(f"Warning: Label file not found for {data_path}")

    return label_paths

# if __name__ == '__main__':
#     path = ['all_data.npy', 'all_ground_truth.npy']
#     images, labels = load_data_coord(path[0], path[1])

if __name__ == '__main__':
    # Load data
    Tk().withdraw()  # Hide the root window
    path_data = askopenfilename(filetypes=[("Data/", "*.npy")], multiple=True)
    label_dir = os.path.dirname(path_data[0]).replace("Processed_data", "Labels").replace("Filtered_data", "Labels")
    path_labels = find_label_paths(path_data, label_dir)

    h = 0
    for i, j in zip(path_labels, path_data):
        if h == 0:
            images, labels = load_data_coord(j, i)
        else:
            vid, vid_lab = load_data_coord(j, i)
            images = np.concatenate((images, vid), axis=0)
            labels = np.concatenate((labels, vid_lab), axis=0)
        h += 1

    normalize = True
    
    if normalize:
        images = images.astype(np.float32) / 255.0
    
    print(f"Loaded images shape: {np.shape(images)}")

    # Get the height and width of the images
    img_height, img_width = images.shape[1], images.shape[2]
    print(f"Image height: {img_height}, Image width: {img_width}")

    # Filter size for the CNN
    filter_size = 10

    # Stride size for the CNN
    stride_size = 2

    input_shape = (img_height, img_width, 1)
    model = build_model(input_shape=input_shape, filter_size=filter_size, stride_size=stride_size)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse", metrics=["mae"])
    
    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    checkpoint_callback = ModelCheckpoint(
        filepath=f'best_CNN_model_filter:{filter_size}_stride={stride_size}_pt2.h5',
        monitor='val_mae',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(monitor='val_mae', patience=30, restore_best_weights=True)
    
    # Track training time
    start_time = time.time()  


    history = model.fit(
        X_train, Y_train,
        epochs=150,
        batch_size=16,
        validation_data=(X_val, Y_val),
        callbacks=[checkpoint_callback, early_stopping],
        verbose=1
    )

    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training took {training_duration*60:.2f} minutes.")

    # Model summary
    model.summary()

    # Load the trained model
    model.load_weights(f'best_CNN_model_filter:{filter_size}_stride={stride_size}_pt2.h5')
    
    # Evaluate on the validation set
    eval_loss, eval_mae = model.evaluate(X_val, Y_val, verbose=1)

    # Predict on validation set
    Y_pred = model.predict(X_val)

    # Denormalize predictions and ground truth
    Y_pred[:,0] *= img_width
    Y_pred[:,1] *= img_height
    Y_val[:,0] *= img_width
    Y_val[:,1] *= img_height

    # Save metrics to a .txt file
    with open("model_metrics.txt", "a") as f:
        f.write(f"\n--- Results from CNN build_model filter size={filter_size} and stride={stride_size} pt. 2 ---\n")
        f.write(f"Loss: {eval_loss:.4f}, MAE: {eval_mae:.4f}\n")
        # f.write("\nSample Predictions (Pixel Space):\n")
        # f.write(f"Predicted: {Y_pred[:10]}\n")
        # f.write(f"Ground Truth: {Y_val[:10]}\n")
        f.write(f"Training Duration: {training_duration/60:.2f} minutes\n")
        f.write("-" * 40 + "\n")

    # Plot training & validation loss and MAE
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"Loss Curve for model with filter size={filter_size} and stride={stride_size}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f"MAE Curve for model with filter size={filter_size} and stride={stride_size}")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Select a random sample for visualization
    idx = np.random.randint(0, len(X_val))
    sample_image = images[idx]
    ground_truth = Y_val[idx]
    prediction = Y_pred[idx]
    
    # Plot image with ground truth and prediction
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(sample_image, cmap='gray')

    plt.scatter(prediction[0], prediction[1], color='red', marker='x', label="Predicted")
    plt.scatter(ground_truth[0], ground_truth[1], color='green', marker='x', label="Ground Truth")

    ax.set_title("Coordinate Prediction vs Ground Truth")
    ax.legend()
    plt.show()





