### CNN-Based Coordinate Regression from Grayscale Images

## Preprocessing.py

This script loads and preprocesses grayscale video data from `.avi` files and extracts corresponding object tracking data from `.trk` files. It combines and saves the processed frames and ground truth coordinates as NumPy arrays for use in downstream tasks like model training or visualization.

#### Features

* **GUI-based file selection** for multiple `.avi` video files
* **Converts videos to grayscale and normalizes pixel values**
* **Optional frame filtering (commented out)**
* **Crops and horizontally flips each frame for consistency**
* **Parses `.trk` XML files to extract ground truth coordinates** using `get_track()` from `MovieFunc.py`
* **Aligns frames and ground truth by frame index**
* **Saves combined data as NumPy arrays**:

  * `all_data_plot.npy` (video frames)
  * `all_ground_truth_plot.npy` (coordinates)

## model_script.py 
This implements a Convolutional Neural Network (CNN) in TensorFlow/Keras for predicting normalized 2D coordinates from grayscale image data.

#### Features
* Interactive file selection for image and label `.npy` files using `tkinter`.
* Data normalization and optional use of residual blocks (did not provide robust results).
* CNN model with customizable filter and stride sizes (filter_size=10 and stride_size=2 yields best results).
* Model training with **early stopping** and **model checkpointing** based on validation MAE (patience=30).
* Visualization of:

  * Training/validation loss and MAE curves
  * Predicted vs. ground truth coordinates on a sample image
* Evaluation metrics and training duration are logged to a `.txt` file called `model_metrics.txt`

#### Model Architecture

* 2 convolutional layers with batch normalization, dropout, and max pooling.
* Fully connected layers followed by a `Dense(2, activation="sigmoid")` output layer for normalized `(x, y)` prediction.
* Optional residual blocks included (commented out, did not yield robust results).

#### Input

* Grayscale images as NumPy arrays (`.npy`)
* Ground truth labels as `(x, y)` coordinates in `.npy` format

#### Output

* Trained model saved as `.h5`
* Evaluation metrics (`loss`, `MAE`) saved to `model_metrics.txt`
* Plot visualizations of training history and prediction sample

## test_plot.py
This script loads a pre-trained convolutional neural network (CNN), to predict 2D coordinates from grayscale video frames. It evaluates performance by comparing predicted coordinates against ground truth tracking data and provides visualization and error metrics.

#### Features

* Loads `.npy` image and label data exported from video/tracking files.
* Applies a trained CNN model to predict coordinates for each frame.
* Calculates and logs **Euclidean prediction errors** per frame.
* Visualizes:

  * Error over time.
  * Prediction vs. ground truth on a random frame.
  * Overlayed playback of video with predicted and ground truth points.

#### Expected Input

* `*.npy` image data (grayscale, normalized/cropped)
* `*.npy` label data (coordinates normalized to \[0, 1])

#### Output

* Error metrics saved in `error_metrics.txt`
* Visualization of prediction error and sample output
* Optionally: playback of predictions over original video

## MovieFunc.py
This script provides interactive tools to **visualize and play back grayscale video frames with overlaid predicted and ground truth coordinates**. It supports frame-by-frame inspection, playback control, and exporting annotated videos.

#### Features

* **Parse `.trk` tracking files** (XML) to extract ground truth coordinates.
* **Interactive video player** with:

  * Overlay of predicted and true coordinates
  * Controls: Play/Pause (`Space`), Step (`A`/`D`), Quit (`Q`)
  * Legends and frame indexing displayed
* **Export annotated videos** with predictions and ground truths overlaid.
* Built using `OpenCV`, `tkinter`, and `NumPy`.

#### Input

* Grayscale video frames stored as a NumPy array (`.npy`)
* Ground truth and predicted coordinate arrays (`.npy` or parsed from `.trk`)

#### Output

* Interactive playback of tracking predictions
* Optional export as `.mp4` with coordinate overlays
