"""Geometry for displaying an entire video frame inside a resizable viewport."""


def fit_video_size(source_size, viewport_size):
    """Fit both dimensions, touching one edge, including upscaling small videos."""
    width, height = source_size
    available_width, available_height = viewport_size
    if min(width, height, available_width, available_height) <= 0:
        raise ValueError('Image and viewport dimensions must be positive')
    if available_width * height <= available_height * width:
        return int(available_width), max(1, int(available_width * height / width))
    return max(1, int(available_height * width / height)), int(available_height)
