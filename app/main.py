from importlib.resources import files
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from make87_messages.core.header_pb2 import Header
import make87 as m87
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned
from make87_messages.geometry.box.boxes_2d_aligned_pb2 import Boxes2DAxisAligned

from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG



class YOLOv10:

    def __init__(self, path: Path, conf_thres: float = 0.2):

        self.conf_threshold = conf_thres

        # Initialize model
        self.session = onnxruntime.InferenceSession(path, providers=onnxruntime.get_available_providers())

        # Get model info
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        return self.process_output(outputs[0])

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
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

        # Rescale boxes to original image dimensions
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
        self.input_height = input_shape[2] if type(input_shape[2]) == int else 640
        self.input_width = input_shape[3] if type(input_shape[3]) == int else 640

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


def main():
    m87.initialize()

    input_topic = m87.get_subscriber(name="IMAGE_DATA", message_type=ImageJPEG)
    output_topic = m87.get_publisher(name="BOUNDING_BOXES", message_type=Boxes2DAxisAligned)
    boxes_entity_name = m87.get_config_value("BOXES_ENTITY_NAME", "boxes", str)
    verbose = m87.get_config_value("VERBOSE", False, bool)

    # Access the 'preprocessor_config.json' file within 'app.hf' package
    yolov10onnx = files("app") / "hf" / "yolov10b.onnx"
    yolov10onnx = Path(str(yolov10onnx))

    detector = YOLOv10(yolov10onnx)

    def callback(message: ImageJPEG):
        jpeg_array = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(jpeg_array, cv2.IMREAD_UNCHANGED)

        class_ids, boxes, confidences = detector(np.array(image))

        boxes2d = Boxes2DAxisAligned(timestamp=message.timestamp, header=m87.header_from_message(Header, message=message, append_entity_path=boxes_entity_name, set_current_time=True))

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            box2d = Box2DAxisAligned(
                x=box[0],
                y=box[1],
                width=box[2] - box[0],
                height=box[3] - box[1],
            )

            boxes2d.boxes.append(box2d)

        output_topic.publish(boxes2d)
        if verbose:
            print(f"Published {len(boxes2d.boxes)} boxes")

    input_topic.subscribe(callback)


if __name__ == "__main__":
    main()
