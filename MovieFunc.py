import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import xml.etree.ElementTree as ET
import time
# from Data_conversion import Data_conversion

# Function to read and parse tracking file
def get_track(trk_path):
    # Parse XML
    tree = ET.parse(trk_path)
    root = tree.getroot()

    tracking_dict = {}

    # Find all <property> elements (frame numbers)
    for prop in root.findall(".//property"):
        frame_number = prop.get("name").strip("[]")  # Extract frame number as string
        if frame_number.isdigit():  # Ensure it's a valid number
            frame_number = int(frame_number)

            # Find <object> inside <property>
            obj = prop.find("object")
            if obj is not None:
                x_elem = obj.find("property[@name='x']")
                y_elem = obj.find("property[@name='y']")

                if x_elem is not None and y_elem is not None:
                    x = float(x_elem.text)  # Convert to float
                    y = float(y_elem.text)  # Convert to float

                    tracking_dict[frame_number] = (int(x), int(y))  # Convert to int

    return tracking_dict

def play_video(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    num_frames = X.shape[0]
    frame_idx = 0
    playing = True

    def draw_frame(frame_idx):
        frame = X[frame_idx].copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for colored markers
        
        # Add a white border for text display
        border_size = 100
        bordered_frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # Draw y_true (red circles)
        cv2.circle(bordered_frame, (int(y_true[frame_idx, 0])+ border_size, int(y_true[frame_idx, 1]) + border_size), 5, (0, 0, 255), -1)
        
        # Draw y_pred (blue crosses)
        cv2.drawMarker(bordered_frame, (int(y_pred[frame_idx, 0]) + border_size, int(y_pred[frame_idx, 1]) + border_size), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        
        # Overlay instructions on the white border
        instructions = [
            "Spacebar: Play/Pause",
            "A: Step Backward",
            "D: Step Forward",
            "Q: Quit"
        ]
        y_offset = 20
        for instr in instructions:
            cv2.putText(bordered_frame, instr, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y_offset += 20
        
        cv2.putText(bordered_frame, f"Frame {frame_idx+1}/{num_frames}", (np.shape(X)[2], 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        legend_x = border_size + 10  # inside the image, not on white border
        legend_y = border_size + 10

        # Draw Ground Truth legend
        cv2.circle(bordered_frame, (legend_x, legend_y), 5, (0, 0, 255), -1)
        cv2.putText(bordered_frame, "RawNet prediction", (legend_x + 15, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw Prediction legend
        legend_y += 25
        cv2.drawMarker(bordered_frame, (legend_x, legend_y), (255, 0, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.putText(bordered_frame, "SticksNet prediction", (legend_x + 15, legend_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        
        cv2.imshow("Video Player", bordered_frame)
    
    while True:
        if playing:
            draw_frame(frame_idx)
            frame_idx = (frame_idx + 1) % num_frames
            time.sleep(0.1)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q' key
            break
        elif key == ord(' '):  # Toggle play/pause on spacebar
            playing = not playing
        elif key == ord('a'):  # Step backward
            frame_idx = (frame_idx - 1) % num_frames
            draw_frame(frame_idx)
        elif key == ord('d'):  # Step forward
            frame_idx = (frame_idx + 1) % num_frames
            draw_frame(frame_idx)
    
    cv2.destroyAllWindows()


def save_movie_with_points(frames, points1, points2, output_filename="output.mp4", fps=30):

    height, width = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    for i, frame in enumerate(frames):
        # Ensure frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)  # Scale and convert to uint8

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert to color
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        cv2.circle(frame, (int(x1), int(y1)), radius=3, color=(0, 0, 255), thickness=-1)  # Draw red point
        cv2.circle(frame, (int(x2), int(y2)), radius=3, color=(255, 0, 0), thickness=-1)  # Draw blue point

        out.write(frame)  # Write frame to video

    out.release()
    print(f"Video saved as {output_filename}")


if __name__ == "__main__":
    X = np.load("all_data2.npy")
    y_true = np.load("all_ground_truth2.npy")
    y_pred = np.zeros_like(y_true)
    play_video(X, y_true, y_pred)
