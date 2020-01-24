import dataclasses
import imgaug.augmentables.bbs as bbs
import numpy as np
import typing as typ

@dataclasses.dataclass(frozen=True)
class DetectedObject:
    label_id: int
    label: str
    score: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_bbox(self) -> bbs.BoundingBox:
        box = bbs.BoundingBox(self.xmin, self.ymin, self.xmax, self.ymax, label=self.label)
        return box

    def to_decoded_array(self, detect_result, num_classes: int) -> typ.List[float]:
        loc = [
            self.xmin / detect_result.image_size[1],
            self.ymin / detect_result.image_size[0],
            self.xmax / detect_result.image_size[1],
            self.ymax / detect_result.image_size[0]
        ]
        cnf = [int(self.label_id == i) for i in range(num_classes)]
        return loc + cnf

    @staticmethod
    def from_decoded_array(decoded_boxes: np.ndarray, classes: typ.List[str], image_size: typ.Tuple[int, int]):
        detected_objs = list()
        for box in decoded_boxes:
            loc = box[0:4]
            class_conf = box[4:]
            class_id = np.argmax(class_conf)
            class_conf = class_conf[class_id]
            class_name = classes[class_id]

            obj = DetectedObject(
                class_id, class_name, class_conf,
                loc[0] * image_size[1],
                loc[1] * image_size[0],
                loc[2] * image_size[1],
                loc[3] * image_size[0]
            )
            detected_objs.append(obj)

        return detected_objs

    @staticmethod
    def from_bbox(box: bbs.BoundingBox, classes: typ.List[str]):
        label_id = classes.index(box.label)
        dec_obj = DetectedObject(label_id, box.label, 1.0, box.x1, box.y1, box.x2, box.y2)
        return dec_obj

@dataclasses.dataclass(frozen=True)
class DetectResult:
    boxes: typ.List[DetectedObject]
    image_size: typ.Tuple[int, int]

    def to_decoded_array(self, classes: typ.List[str]):
        bboxes = [item.to_decoded_array(self, classes) for item in self.boxes]
        bboxes = np.array(bboxes)
        return bboxes

    @staticmethod
    def from_decoded_array(decoded_boxes: np.ndarray, classes: typ.List[str], image_size: typ.Tuple[int, int]):
        bboxes = DetectedObject.from_decoded_array(decoded_boxes, classes, image_size)
        det_res = DetectResult(bboxes, image_size)
        return det_res

    @staticmethod
    def from_bbox_on_image(bbox_on_image: bbs.BoundingBoxesOnImage, classes: typ.List[str]):
        bboxes = [DetectedObject.from_bbox(box, classes) for box in bbox_on_image.bounding_boxes]
        image_size = (bbox_on_image.height, bbox_on_image.width)
        det_res = DetectResult(bboxes, image_size)
        return det_res
