"""
OccluSynth — Occlusion-Aware 3D Scene Reconstruction.

Package layout:
    occlusynth.data            — ScanNet dataset I/O, anchor sampling
    occlusynth.models          — VGGT-Omega wrapper, depth calibration, adapter
    occlusynth.fusion          — TSDF volume integration
    occlusynth.viz             — depth colourisation, Rerun viewer
    occlusynth.utils           — device selection, path helpers
"""

__version__ = "0.1.0"
__author__  = "Aditya Agarwal"
