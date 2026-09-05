"""geo - camera geometry from the pinhole model to metric stereo depth.

The package is deliberately split along the seams of the geometry rather than
along the seams of the OpenCV API, because the seams of the geometry are what a
reader has to hold in their head:

    pinhole     a 3-D point in the CAMERA frame -> a pixel      (K, [R|t])
    distortion  the lens correction that sits between the two   (Brown-Conrady)
    synthetic   scenes with known ground truth to test against
    homography  plane-to-plane maps, the linear estimate (DLT)
    calibrate   many views of a known board -> K and D
    epipolar    two views of an unknown scene -> F, E
    stereo      two rectified views -> disparity
    depth       disparity -> metres, and how wrong those metres are
"""

__all__ = [
    "pinhole",
    "distortion",
    "synthetic",
    "homography",
    "calibrate",
    "epipolar",
    "stereo",
    "depth",
]

__version__ = "1.0.0"
