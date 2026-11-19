# Yolo-rest
Contains a service that allows you to run ultralytics yolo on your cpu(not recommended) or coral tpu the api has the same request and response as deepstack, this enables the usage with frigate.

## Docker compose:
```yaml
networks:
  yolorest:
    driver: bridge
    internal: true
    driver_opts:
      com.docker.network.bridge.name: yolorest-dsp

services:
  frigate:
    image: ghcr.io/blakeblackshear/frigate:${FRIGATE_VERSION}
    container_name: frigate
    restart: on-failure
    stop_grace_period: 30s
    mem_limit: 1.5G
    cpus: 2
    shm_size: "512mb"
    group_add:
      - "${GROUP_RENDER}"
      - "${GROUP_VIDEO}"
      - "${GROUP_INPUT}"
    security_opt:
      - "no-new-privileges=true"
    cap_add:
      - CAP_PERFMON
    networks:
      yolorest:
    environment:
      - LIBVA_DRIVER_NAME=i965
      - TZ=Europe/Stockholm
    volumes:
      - /etc/localtime:/etc/localtime:ro
      - ./frigate/config:/config
      - /storage/surveillance/frigate/media:/media/frigate
      - /storage/surveillance/frigate/cache:/tmp/cache
    devices:
      - /dev/dri/card1:/dev/dri/card1
      - /dev/dri/renderD128:/dev/dri/renderD128
    depends_on:
      yolorest:
        condition: service_healthy
        restart: true
    labels:
      - "org.label-schema.group=Surveillance"

  yolorest:
    image: ghcr.io/anderssonpeter/yolorest:${YOLOREST_VERSION}
    container_name: yolorest
    restart: on-failure
    mem_limit: 256M
    cpus: 0.5
    user: "${USER_SURVEILLANCE}:${GROUP_SURVEILLANCE}"
    read_only: true
    group_add:
      - "${GROUP_CORAL}"
    security_opt:
      - "no-new-privileges=true"
    networks:
      yolorest:
    environment:
      - TZ=Europe/Stockholm
    volumes:
      - /etc/localtime:/etc/localtime:ro
      - ./yolorest/models:/models
    devices:
      - /dev/apex_0:/dev/apex_0
    command:
      - "--device=pci"
      - "--label_file=/models/labelmap_yolov8.txt"
      - "--model_file=/models/yolov8m_320_edgetpu.tflite" # This model is slow about 90-100ms, not recommended if you run more than one camera
    labels:
      - "org.label-schema.group=Surveillance"
```

## Frigate config
```yaml
detectors:
  yolo-rest:
    type: deepstack
    api_url: http://yolorest:8000/detect
    api_timeout: 0.18 # Only set this if you use a slow model

model:
  labelmap_path: /models/labelmap_yolov8.txt
```


# notes
Ultralytics model source: https://github.com/ultralytics/assets/releases/tag/v8.3.0


## Convert model:
`docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu yolo export model=/models/[name].pt format=[edgetpu/tflite int8]`


## Optimized YOLOv9 model

Support for the modified YOLOv9 model from
[frigate-detector-edgetpu-yolo9](https://github.com/dbro/frigate-detector-edgetpu-yolo9) is enabled with
`--model_type optimized_yolov9`. Download the model from the
[v2.1 release](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/download/v2.1/yolov9-s-relu6-sl21_320_int8_edgetpu.tflite).

```bash
python main.py \
  --model_type optimized_yolov9 \
  --model_file /models/yolov9-s-relu6-sl21_320_int8_edgetpu.tflite \
  --label_file /models/labelmap_yolov9.txt \
  --device pci
```

The label file must match the model's class order and use the same `id name` format as the other label files,
for example `0 person`. The `labels-coco17.txt` file from the reference repository is a plain list without
class IDs, so prefix each line with its zero-based index before using it here.

The EdgeTPU-compiled model must run on a Coral device (`--device pci` or `--device usb`); `--device cpu` cannot
run it. The default remains `--model_type yolov8` when the flag is omitted.

## Dev load vent
`source ~/object-detection/bin/activate`