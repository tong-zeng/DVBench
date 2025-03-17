import cv2
import numpy as np


def read_video(video_path):
    """
    Reads a video file and returns its frames as a numpy array.

    Args:
        video_path (str): Path to the video file.

    Returns:
        numpy.ndarray: A 4D numpy array of shape (num_frames, height, width, channels).
    """
    cap = cv2.VideoCapture(video_path)  # Initialize video capture
    # Ensure the video capture is opened
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get frame rate
    if frame_rate == 0:  # Try another method if FPS is not available
        raise ValueError(f"Error opening video file: {video_path}")

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # Append the frame to the list

    cap.release()  # Release the video capture
    return np.array(frames)  # Convert list of frames to a numpy array
