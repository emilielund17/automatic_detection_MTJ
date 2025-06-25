import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable
import sklearn
from Preprocessing import Data_conversion
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PModel import load_data, build_cnn, build_model
from model_script import load_data_coord


# def track_object(model_path, img, y_true = None):
#     model = tf.keras.models.load_model(model_path)
#     if len(img.shape) == 2:
#         img = np.expand_dims(img, axis=-1)
#     img = np.expand_dims(img, axis=0)
#     prediction = model.predict(img)[0]
    
#     # Denormalize the bounding box coordinates
#     _, h, w, _ = np.shape(img)
#     x, y, width, height = prediction
#     x = int(x * w)
#     y = int(y * h)
#     y_true += [width // 2, height // 2, 0, 0]
#     y_true *= [X.shape[2], X.shape[1], X.shape[2], X.shape[1]]
#     y_true = [int(i) for i in y_true]
#     width = int(width * w)
#     height = int(height * h)
    
#     return (x, y, width, height, img, y_true)

# def plot_tracking(image, x, y, width, height, x_true = [], y_true = []):
#     fig, ax = plt.subplots(1)
#     ax.imshow(image[0,:,:], cmap='gray')
#     ax.add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor='red', lw=2))
#     if not x_true is None and not y_true is None:
#         ax.plot(x_true, y_true, 'ro')
#     plt.show()

if __name__ == '__main__':
    X, y_true = load_data_coord("all_data.npy", "all_ground_truth.npy")
    model = build_model(X.shape[1:])
    model.load_weights("best_model_coordinate.h5")

    y_pred = np.zeros_like(y_true)
    for i in range(X.shape[0]):
        y_pred[i] = model.predict(X[i,:,:])
    y_delta = y_true - y_pred

    frame_idx = range(0, X.shape[0])
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax[0].scatter(y_delta[0], frame_idx, color='red', marker='x', label="The error in x")
    ax[0].scatter(y_delta[1], frame_idx, color='blue', marker='x', label="The error in y")
    ax[1].legend(["x", "y"])
    ax[1].hist(y_delta[0], bins=50, color='red')
    ax[1].hist(y_delta[1], bins=50, color='blue')
    ax[0].legend(["x", "y"])

    plt.show()
    
    