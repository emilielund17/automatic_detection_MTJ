import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from model_script import build_model, load_data_coord, find_label_paths
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from MovieFunc import play_video
import time

# Compute Euclidean distance between predicted and ground truth points
def compute_error(predictions, ground_truth, img_width, trans_length=60):
    errors = np.linalg.norm(predictions - ground_truth, axis=1) 
    errors = (trans_length / img_width) * errors # Compute distance frame-wise
    return errors

# Plot error per frame
def plot_error(errors):
    max = np.max(errors)
    plt.figure(figsize=(10, 5))
    plt.plot(errors, marker="o", linestyle="-", color="red")
    plt.xlabel("Frame Index")
    plt.ylabel("Prediction Error (mm)")
    plt.ylim([0, max])
    plt.title("Error per Frame")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    
    # # Load new dataset for predictions
    # new_data_path = "all_data_plot.npy"  # Update with correct path
    # ground_truth_path = "all_ground_truth_plot.npy" 
    # X, Y_true = load_data_coord(new_data_path, ground_truth_path)
    # print(X.shape, Y_true.shape)

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
    
    # normalize = True
    
    # if normalize:
    #     images = images.astype(np.float32) / 255.0
    
    print(f"Loaded images shape: {np.shape(images)}")
    print(f"mean: {np.mean(images)}, std: {np.std(images)}")

    # Define model parameters
    img_height, img_width = images.shape[1], images.shape[2]
    filter_size = 10
    stride_size = 2
    input_shape = (img_height, img_width, 1)
    model = build_model(input_shape=input_shape, filter_size=filter_size, stride_size=stride_size)

    # Load trained weights
    model.load_weights(f"best_CNN_model_filter:{filter_size}_stride={stride_size}_filtered_pt5.h5")
    print("Model weights loaded successfully.")
    
    start_time = time.time()
    # Make predictions
    Y_pred = model.predict(images)
    end_time = time.time()
    prediction_duration = end_time - start_time
    print(f"Prediction took {prediction_duration:.2f} seconds.")

    # Convert predictions back to pixel space
    Y_pred[:, 0] *= img_width
    Y_pred[:, 1] *= img_height
    labels[:, 0] *= img_width
    labels[:, 1] *= img_height
    # print(f'Prediction: {Y_pred[:10]} \n Ground Truth: {labels[:10]}')
    # Compute error per frame
    errors = compute_error(Y_pred, labels, img_width=img_width)

        # Save metrics to a .txt file
    with open("error_metrics.txt", "a") as f:
        f.write(f"\n--- Error metrics for RawNet model ---\n")
        f.write(f"mean error: {np.mean(errors)}\n")
        f.write(f"std error: {np.std(errors)}\n")
        f.write(f"max error: {np.max(errors)}\n")
        f.write(f"min error: {np.min(errors)}\n")
        f.write("-" * 50 + "\n")
    
    # Plot errors
    plot_error(errors)


    idx = np.random.randint(0, len(images))
    print(f"Random sample index: {idx}")
    sample_image = images[idx]
    ground_truth = labels[idx]
    prediction = Y_pred[idx]
    
    # Plot image with ground truth and prediction
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(sample_image, cmap='gray')

    plt.scatter(prediction[0], prediction[1], color='red', marker='x', label="Predicted")
    plt.scatter(ground_truth[0], ground_truth[1], color='green', marker='x', label="Ground Truth")

    ax.set_title("SticksNet - Coordinate Prediction vs Ground Truth for random sample")
    ax.legend()
    plt.show()

    # Play video with predictions
    play_video(images, labels, Y_pred)
