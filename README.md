# ArUco Marker Displacement Measurement

Measure displacement of Aruco Marker in euclidean distance from 2 consecutive images given camera instrinsics.
In layman terms, this repo is a Digital Image Correlation (DIC) of ArUco Marker.

## Applications:
applicable to measure anything which the marker can be pasted there, and the local deflection across the marker is negligible.
example:
- Bridge deflection measurement

## Limitations of this version:
- 3DOFs only: only X-Y axises (marker plane) movement and rotation on the Z axis of marker are allowed.
- The marker plane should be exactly perpendicular to the camera's principle axis (or even coincide, if you're luck enough!)
- Unoptimized corner location detection (I made some noise reduction

## Next milestone:
- perspective modelling: this will unlock 6DOFs movement

## About this project
this project is intended as my senior thesis during my undergraduate study on Institut Teknologi Bandung.
