from dvbench.visual_processing.video_processing_interface import VideoProcessingInterface
import cv2
from PIL import Image
import numpy as np


class Video(VideoProcessingInterface):
    def __init__(self, video_path):
        """
        Initializes the VideoProcessor object with the given video path.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)  # Initialize video capture
        # Ensure the video capture is opened
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))  # Get frame rate
        if self.frame_rate == 0:  # Try another method if FPS is not available
            raise ValueError(f"Error opening video file: {video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frame count
        self.duration = self.frame_count / self.frame_rate  # Calculate video duration

    def extract_frames_by_interval(self, frame_interval):
        """
        Extracts frames from the video at a certain interval.

        Args:
            frame_interval (int): Interval between frames to be extracted.

        Returns:
            list: List of PIL Image objects for the extracted frames.
        """
        frames = []
        count = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()  # Read the next frame
            if not ret:
                break
            if count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                pil_image = Image.fromarray(frame_rgb)  # Convert to PIL Image
                frames.append(pil_image)  # Append the frame to the list if it matches the interval
            count += 1
        return FrameSequence(frames)

    def extract_frames_by_fps(self, fps):
        """
        Extracts frames from the video at a specified frames per second (FPS) rate.

        Args:
            fps (float): Number of frames to extract per second.

        Returns:
            list: List of PIL Image objects for the extracted frames.
        """
        interval = int(self.frame_rate / fps)
        return self.extract_frames_by_interval(interval)

    def extract_evenly_spaced_frames(self, num_frames):
        """
        Extracts a specified number of evenly spaced frames from the video.

        Args:
            num_frames (int): Number of frames to extract.

        Returns:
            list: List of PIL Image objects for the extracted frames.
        """
        frames = []
        step = max(1, self.frame_count // num_frames)
        for i in range(0, self.frame_count, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set the current frame position
            ret, frame = self.cap.read()  # Read the frame at the current position
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            pil_image = Image.fromarray(frame_rgb)  # Convert to PIL Image
            frames.append(pil_image)
            if len(frames) >= num_frames:
                break
        return FrameSequence(frames)

    def split_into_clips(self, time_points):
        """
        Splits the video into clips at specified time points.

        Args:
            time_points (list): List of time points (in seconds) to split the video.

        Returns:
            list: List of FramesProcessor objects for the video clips.
        """
        clips = []
        for i in range(len(time_points) - 1):
            start_time = time_points[i]
            end_time = time_points[i + 1]
            start_frame = int(start_time * self.frame_rate)
            end_frame = int(end_time * self.frame_rate)

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            clip_frames = []
            for j in range(start_frame, end_frame):
                ret, frame = self.cap.read()
                if not ret:
                    break
                clip_frames.append(frame)

            if clip_frames:
                clips.append(FrameSequence(clip_frames, self.frame_rate))

        return clips

    def get_video_info(self):
        """
        Retrieves information about the video.

        Returns:
            dict: Dictionary containing video information including path, frame rate, frame count, and duration.
        """
        info = {
            'path': self.video_path,
            'frame_rate': self.frame_rate,
            'frame_count': self.frame_count,
            'duration': self.duration
        }
        return info

    def crop(self, x, y, width, height):
        """
        Crops the video to the specified rectangle. If width or height are negative, they are treated as percentages.

        Args:
            x (int): The x-coordinate of the top-left corner of the rectangle.
            y (int): The y-coordinate of the top-left corner of the rectangle.
            width (int): The width of the rectangle. If <= 1.0, treated as a percentage of the original width.
            height (int): The height of the rectangle. If <= 1.0, treated as a percentage of the original height.

        Returns:
            list: List of numpy arrays representing the cropped frames.
        """
        if width < 0 or height < 0:
            raise ValueError("Width and height must not be negative.")

        # Get the original width and height of the video
        original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the actual width and height if they are given as percentages
        if width <= 1:
            width = int(original_width * width)
        if height <= 1:
            height = int(original_height * height)

        cropped_frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            cropped_frame = frame[y:y + height, x:x + width]
            cropped_frames.append(cropped_frame)
        return FrameSequence(cropped_frames)

    def __enter__(self):
        """
        Enters the runtime context related to this object.

        Returns:
            Video: The instance of the VideoProcessor.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the runtime context related to this object, releasing the video capture.

        Args:
            exc_type: The exception type.
            exc_value: The exception value.
            traceback: The traceback object.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()  # Release the video capture

    def to_np_ndarray(self):
        """
        Converts the video to a numpy array.

        Returns:
            numpy.ndarray: A 4D numpy array of shape (num_frames, height, width, channels).
        """
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)  # Append the frame to the list

        return np.array(frames)  # Convert list of frames to a numpy array


class FrameSequence(VideoProcessingInterface):

    def __init__(self, frames, frame_rate=1):
        """
        Initializes the FramesProcessor object with the given frames and frame rate.

        Args:
            frames (list): List of frames (numpy arrays).
            frame_rate (float): Frame rate of the video.
        """
        self.frames = frames
        self.frame_rate = frame_rate
        self.frame_count = len(frames)
        self.duration = self.frame_count / self.frame_rate

    def extract_frames_by_interval(self, frame_interval):
        """
        Extracts frames from the frames list at a certain interval.

        Args:
            frame_interval (int): Interval between frames to be extracted.

        Returns:
            list: List of PIL Image objects for the extracted frames.
        """
        frames = []
        for i in range(0, len(self.frames), frame_interval):
            frame_rgb = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            pil_image = Image.fromarray(frame_rgb)  # Convert to PIL Image
            frames.append(pil_image)
        return frames

    def extract_frames_by_fps(self, fps):
        """
        Extracts frames from the frames list at a specified frames per second (FPS) rate.

        Args:
            fps (float): Number of frames to extract per second.

        Returns:
            list: List of PIL Image objects for the extracted frames.
        """
        interval = int(self.frame_rate / fps)
        return self.extract_frames_by_interval(interval)

    def extract_evenly_spaced_frames(self, num_frames):
        """
        Extracts a specified number of evenly spaced frames from the frames list.

        Args:
            num_frames (int): Number of frames to extract.

        Returns:
            list: List of PIL Image objects for the extracted frames.
        """
        frames = []
        step = max(1, len(self.frames) // num_frames)
        for i in range(0, len(self.frames), step):
            frame_rgb = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            pil_image = Image.fromarray(frame_rgb)  # Convert to PIL Image
            frames.append(pil_image)
            if len(frames) >= num_frames:
                break
        return frames

    def split_into_clips(self, time_points):
        """
        Splits the frames list into clips at specified time points.

        Args:
            time_points (list): List of time points (in seconds) to split the frames list.

        Returns:
            list: List of FramesProcessor objects for the frame clips.
        """
        clips = []
        start_frame = 0
        for end_time in time_points[1:]:
            end_frame = int(end_time * self.frame_rate)
            clip_frames = self.frames[start_frame:end_frame]
            clips.append(FrameSequence(clip_frames, self.frame_rate))
            start_frame = end_frame
        return clips

    def crop(self, x, y, width, height):
        """
        Crops the video to the specified rectangle.

        Args:
            x (int): The x-coordinate of the top-left corner of the rectangle.
            y (int): The y-coordinate of the top-left corner of the rectangle.
            width (int): The width of the rectangle.
            height (int): The height of the rectangle.

        Returns:
            list: List of numpy arrays representing the cropped frames.
        """
        if width < 0 or height < 0:
            raise ValueError("Width and height must not be negative.")
        cropped_frames = []
        for frame in self.frames:
            original_height, original_width = frame.size
            # Calculate actual width and height if negative values are given
            if width <= 1.0:
                width = int(original_width * width)
            if height <= 1.0:
                height = int(original_height * height)

            cropped_frame = frame.crop((x, y, x + height, y + width))
            cropped_frames.append(cropped_frame)
        return FrameSequence(cropped_frames, self.frame_rate)

    def get_video_info(self):
        """
        Retrieves information about the frames list.

        Returns:
            dict: Dictionary containing frames information including frame rate, frame count, and duration.
        """
        info = {
            'frame_rate': self.frame_rate,
            'frame_count': self.frame_count,
            'duration': self.duration
        }
        return info

    def __len__(self):
        """
        Returns the number of frames in the FrameSequence.

        Returns:
            int: Number of frames.
        """
        return self.frame_count


if __name__ == "__main__":
    # Usage Example

    input_video_path = "../../videos/2834107.mp4"

    with Video(input_video_path) as video_processor:
        # Extract frames at every 30th frame
        interval_frames = video_processor.extract_frames_by_interval(30)
        print(f"Extracted {len(interval_frames)} frames by interval.")

        # Extract frames at 5 frames per second
        fps_frames = video_processor.extract_frames_by_fps(5)
        print(f"Extracted {len(fps_frames)} frames at 5 FPS.")

        # Extract 10 evenly spaced frames from the video
        evenly_spaced_frames = video_processor.extract_evenly_spaced_frames(10)
        print(f"Extracted {len(evenly_spaced_frames)} evenly spaced frames.")

        # Split video into clips at specified time points
        time_points = [0, 10, 20, 30]  # Example time points in seconds
        clips = video_processor.split_into_clips(time_points)
        print(f"Video split into {len(clips)} clips at time points: {time_points}")

        # Process the first clip with another FramesProcessor instance
        # first_clip: FramesClip = clips[0]
        # first_clip_processor = FramesProcessor(first_clip.frames, first_clip.frame_rate)
        first_clip_processor = clips[0]
        first_clip_info = first_clip_processor.get_video_info()
        print("First clip info:", first_clip_info)

        # Extract frames at every 5th frame from the first clip
        first_clip_interval_frames = first_clip_processor.extract_frames_by_interval(5)
        print(f"Extracted {len(first_clip_interval_frames)} frames by interval from the first clip")

        # Get video information
        info = video_processor.get_video_info()
        print(info)

    # The video capture will be released automatically when the with block exits
