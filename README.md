Ultralytics model source: https://github.com/ultralytics/assets/releases/tag/v8.3.0


# Convert model:
`docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu yolo export model=/models/[name].pt format=[edgetpu/tflite int8]`


# Dev load vent
`source ~/object-detection/bin/activate`