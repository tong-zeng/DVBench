from abc import ABC, abstractmethod


class VideoProcessingInterface(ABC):
    @abstractmethod
    def crop(self, x, y, width, height):
        pass

    @abstractmethod
    def extract_frames_by_interval(self, frame_interval):
        pass

    @abstractmethod
    def extract_evenly_spaced_frames(self, num_frames):
        pass

    @abstractmethod
    def split_into_clips(self, time_points):
        pass

    @abstractmethod
    def get_video_info(self):
        pass