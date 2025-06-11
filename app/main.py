import json
from importlib.resources import files
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
import logging
from make87.config import load_config_from_env
from make87.encodings import ProtobufEncoder
from make87.interfaces.zenoh import ZenohInterface
import threading
from make87_messages.core.header_pb2 import Header
from make87_messages.detection.box.box_2d_pb2 import Box2DAxisAligned
from make87_messages.detection.box.boxes_2d_pb2 import Boxes2DAxisAligned
from make87_messages.detection.ontology.model_ontology_pb2 import ModelOntology
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned as Box2DAxisAlignedGeometry
from make87_messages.image.uncompressed.any_pb2 import ImageRawAny


class YOLOv10:
    def __init__(self, path: Path, conf_thres: float = 0.2, provider="CPUExecutionProvider"):
        self.conf_threshold = conf_thres

        available_providers = onnxruntime.get_available_providers()
        # Prefer CUDA if available, fallback to CPU
        if provider == "CUDAExecutionProvider" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        logging.info(f"Available providers: {available_providers}, using: {providers}")
        self.session = onnxruntime.InferenceSession(str(path), providers=providers)

        self.get_input_details()
        self.get_output_details()

    def __call__(self, image: np.ndarray):
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        return self.process_output(outputs[0])

    def prepare_input(self, image: np.ndarray):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        output = output.squeeze()
        boxes = output[:, :-2]
        confidences = output[:, -2]
        class_ids = output[:, -1].astype(int)
        mask = confidences > self.conf_threshold
        boxes = boxes[mask, :]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        boxes = self.rescale_boxes(boxes)
        return class_ids, boxes, confidences

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape
        self.input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
        self.input_width = input_shape[3] if isinstance(input_shape[3], int) else 640

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


def extract_image_from_raw_any(msg: ImageRawAny) -> np.ndarray:
    """Extracts a numpy image from ImageRawAny message, converting YUV to BGR if needed."""
    if msg.HasField("rgb888"):
        arr = np.frombuffer(msg.rgb888.data, dtype=np.uint8).reshape((msg.rgb888.height, msg.rgb888.width, 3))
        return arr
    elif msg.HasField("rgba8888"):
        arr = np.frombuffer(msg.rgba8888.data, dtype=np.uint8).reshape((msg.rgba8888.height, msg.rgba8888.width, 4))
        return arr
    elif msg.HasField("yuv420"):
        h, w = msg.yuv420.height, msg.yuv420.width
        yuv = np.frombuffer(msg.yuv420.data, dtype=np.uint8)
        # YUV420p: Y plane (h*w), U (h/2*w/2), V (h/2*w/2)
        yuv = yuv.reshape((h * 3 // 2, w))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return bgr
    elif msg.HasField("yuv422"):
        h, w = msg.yuv422.height, msg.yuv422.width
        yuv = np.frombuffer(msg.yuv422.data, dtype=np.uint8)
        # YUV422p: Y plane (h*w), U (h*w/2), V (h*w/2)
        yuv = yuv.reshape((h * 2, w))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_Y422)
        return bgr
    elif msg.HasField("yuv444"):
        h, w = msg.yuv444.height, msg.yuv444.width
        yuv = np.frombuffer(msg.yuv444.data, dtype=np.uint8)
        # YUV444p: Y plane (h*w), U (h*w), V (h*w)
        yuv = yuv.reshape((h * 3, w))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr
    else:
        logging.error("No supported image format found in ImageRawAny")
        raise ValueError("No supported image format found in ImageRawAny")


def main():
    application_config = load_config_from_env()
    image_raw_any_encoder = ProtobufEncoder(ImageRawAny)
    ontology_encoder = ProtobufEncoder(ModelOntology)
    boxes_encoder = ProtobufEncoder(Boxes2DAxisAligned)
    zenoh_interface = ZenohInterface("zenoh-client")

    # Configuration values
    boxes_entity_name = application_config.config.get("BOXES_ENTITY_NAME", "boxes")
    conf_threshold = float(application_config.config.get("CONFIDENCE_THRESHOLD", 0.6))

    # Providers / Subscribers / Publishers
    ontology_provider = zenoh_interface.get_provider("MODEL_ONTOLOGY")
    try:
        raw_any_subscriber = zenoh_interface.get_subscriber("IMAGE_DATA")
    except KeyError:
        raw_any_subscriber = None
        logging.warning("No subscriber found for IMAGE_DATA. Skipping subscriber loop.")
    detections_publisher = zenoh_interface.get_publisher("BOUNDING_BOXES")
    detections_provider = zenoh_interface.get_provider("DETECTIONS")

    # Load model and config
    yolov10onnx = Path(files("app") / "hf" / "yolov10b.onnx")
    yolov10config = Path(files("app") / "hf" / "config.json")
    detector = YOLOv10(yolov10onnx, conf_thres=conf_threshold)
    with open(yolov10config) as f:
        config = json.load(f)

    # Ontology provider
    def ontology_callback():
        header = Header()
        header.timestamp.GetCurrentTime()
        class_entries = [
            ModelOntology.ClassEntry(
                id=int(class_id),
                label=class_label,
            )
            for class_id, class_label in config["id2label"].items()
        ]
        return ModelOntology(header=header, classes=class_entries)

    def serve_ontology():
        while True:
            with ontology_provider.recv() as query:
                response = ontology_callback()
                response_encoded = ontology_encoder.encode(response)
                query.reply(key_expr=query.key_expr, payload=response_encoded)

    # Detection provider
    def detections_callback(message: ImageRawAny) -> Boxes2DAxisAligned:
        image = extract_image_from_raw_any(message)
        # Always pass BGR or RGB image to detector
        class_ids, boxes, confidences = detector(np.array(image))
        header = Header()
        header.CopyFrom(message.header)
        header.entity_path = message.header.entity_path + f"/{boxes_entity_name}"
        boxes2d = Boxes2DAxisAligned(
            header=header,
            boxes=[
                Box2DAxisAligned(
                    geometry=Box2DAxisAlignedGeometry(
                        header=header,
                        x=box[0],
                        y=box[1],
                        width=box[2] - box[0],
                        height=box[3] - box[1],
                    ),
                    confidence=float(confidence),
                    class_id=int(class_id),
                )
                for box, class_id, confidence in zip(boxes, class_ids, confidences)
            ],
        )
        return boxes2d

    def serve_detections():
        while True:
            with detections_provider.recv() as query:
                request = image_raw_any_encoder.decode(query.payload.to_bytes())
                response = detections_callback(request)
                response_encoded = boxes_encoder.encode(response)
                query.reply(key_expr=query.key_expr, payload=response_encoded)

    ontology_thread = threading.Thread(target=serve_ontology)
    detections_thread = threading.Thread(target=serve_detections)
    ontology_thread.start()
    detections_thread.start()

    try:
        if raw_any_subscriber is not None:
            for sample in raw_any_subscriber:
                msg = image_raw_any_encoder.decode(sample.payload.to_bytes())
                detections_publisher.put(
                    payload=boxes_encoder.encode(detections_callback(msg))
                )
    finally:
        ontology_thread.join()
        detections_thread.join()


if __name__ == "__main__":
    main()
