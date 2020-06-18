# ArUco Marker Displacement Measurement

Measure displacement of Aruco Marker in euclidean distance from 2 consecutive images given camera instrinsics.
In layman terms, this repo is a Digital Image Correlation (DIC) of ArUco Marker.

## Applications:
applicable to measure anything which the marker can be pasted there, and the local deflection across the marker is negligible.
example:
- Bridge deflection measurement

## Assumptions:
- your camera is really, really static. Very slight movement will interfere the calculation (if you admitted that you can't make it really static, there is a stabilizer.py for make your image looks static for better calculation, assuming there is only slight movement)

## Limitations of this version:
- 3DOFs only: only X-Y axises (marker plane) translational movement and rotational movement on the Z axis of marker are allowed.
- The marker plane should be exactly perpendicular to the camera's principle axis (or even coincide, if you're luck enough!)
- Unoptimized corner location detection (I made some noise reduction

## Next milestone:
- perspective modelling: this will unlock 6DOFs movement

## About this project
this project was intended as my undergraduate thesis at Institut Teknologi Bandung
