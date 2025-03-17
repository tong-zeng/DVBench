import av
import numpy as np


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format='rgb24') for x in frames])


def pyav_extract_frames_by_fps(video_path: str, target_fps: int = 8) -> np.ndarray:
    """Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to video file
        target_fps: Number of frames to extract per second (default: 8)
    
    Returns:
        np.ndarray: Stack of extracted frames in RGB format
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    
    # Calculate frame indices based on target FPS
    avg_rate = video_stream.average_rate or video_stream.rate  # fallback to base rate if average_rate is None
    if avg_rate is None:
        raise ValueError(f"Video {video_path} has no valid frame rate information")
    
    video_duration = video_stream.frames / avg_rate  # duration in seconds
    num_frames = int(video_duration * target_fps)
    if num_frames == 0:
        raise ValueError(f"Video {video_path} has no frames to extract")
    
    if num_frames > video_stream.frames:
        raise ValueError(f"Video {video_path} has fewer frames than requested ({num_frames})")
    
    # Sample uniformly num_frames frames from the video
    indices = np.linspace(0, video_stream.frames - 1, num_frames).astype(int)
    np_frames = read_video_pyav(container, indices)
    return np_frames