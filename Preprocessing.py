import os
import time
import cv2
import numpy as np
from MovieFunc import get_track
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

def Data_conversion(path, filter=False, filter_size=5):
    # Debug: Check if the file exists
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
    else:
        print(f"File found at {path}")

    # Load video file
    cap = cv2.VideoCapture(path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break  # Stop if video ends or an error occurs

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[43:300, 265:904] # image is [43:605, 265:904]    

        frame = frame.astype(np.float32) / 255.0
        frame = cv2.flip(frame, 1)
        frames.append(frame)

    cap.release()
    frames_tensor = np.array(frames)

    return frames_tensor


if __name__ == '__main__':
    
    # Open file dialog to select an AVI file
    Tk().withdraw()  # Hide the root window
    file_path = askopenfilename(filetypes=[("Data/", "*.avi")], multiple=True)

    all_data = np.array([])
    all_ground_truth = np.array([])

    for i, file_name in enumerate(file_path):
        
        trk_path = file_name[:-3] + "trk"
        tracking_dict = get_track(trk_path)
        ground_truth = np.array(list(tracking_dict.values()))
        ground_truth = ground_truth - [264, 42]  # image is [43:605, 265:904]
        
        start = time.time()
        vid_data = Data_conversion(file_name, filter=False, filter_size=51)
        end = time.time()
        print(f"Time taken to process video: {end - start} seconds")

        print(f"The video data was of the dimensions: {np.shape(vid_data)}")
        print(f"The ground truth data was of the dimensions: {np.shape(ground_truth)}")

        if vid_data.shape[0] != len(ground_truth):
            vid_data = vid_data[list(tracking_dict.keys()), :, :]
            print(f'New video data shape: {np.shape(vid_data)}')
        
        if len(all_data) == 0:
            all_data = vid_data
            all_ground_truth = ground_truth
        else:
            all_data = np.concatenate((all_data, vid_data), axis=0)
            all_ground_truth = np.concatenate((all_ground_truth, ground_truth), axis=0)

        print(f"Total data shape: {np.shape(all_data)}")
        print(f"Total ground truth shape: {np.shape(all_ground_truth)}")

    np.save("all_data_plot", all_data)
    np.save("all_ground_truth_plot", all_ground_truth)

