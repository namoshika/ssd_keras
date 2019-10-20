"""Some utils for SSD."""

import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image

class BBoxUtility(object):
    """Utility class to do some stuff with bounding boxes and priors.

    # Arguments
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold.
        top_k: Number of total bboxes to be kept per image after nms step.

    # References
        https://arxiv.org/abs/1512.02325
    """
    # TODO add setter methods for nms_thresh and top_K
    def __init__(self, classes, priors, overlap_threshold=0.5, nms_thresh=0.45, top_k=400):
        self.classes = classes
        self.priors = priors
        self.overlap_threshold = overlap_threshold
        self.nms_thresh = nms_thresh
        self.top_k = top_k

    def iou(self, box):
        """Compute intersection over union for the box with all priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).

        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_priors).
        """
        # compute intersection
        inter_ul = np.maximum(self.priors[:, 0:2], box[:2])
        inter_br = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = np.maximum(inter_br - inter_ul, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def encode_box(self, boundingbox, return_iou=True):
        """Encode box for training, do it only for assigned priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.

        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """
        # assign priors
        # shape: (num_priors,8) 8: [x_min, y_min, x_max, y_max, var1, var2, var3, var4]
        iou = self.iou(boundingbox)
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        # memo: 全座標・サイズの単位は比率
        defaultbox_xyxy = self.priors[assign_mask]
        defaultbox_xyxy = defaultbox_xyxy[:, 0:4]
        defaultbox_centerxy = (defaultbox_xyxy[:, 0:2] + defaultbox_xyxy[:, 2:4]) / 2
        defaultbox_wh = defaultbox_xyxy[:, 2:4] - defaultbox_xyxy[:, 0:2]
        variances = self.priors[assign_mask, 4:8]

        # memo: encoded_box is near equal predictions in decode_boxes
        # shape: (num_priors, 4 + 1) 4: [x_min, y_min, x_max, y_max] 1: [iou]
        # solve: encoded_box[assign_mask, 0:2]
        #   (Take expression from decode_boxes)
        #   bbox_centerxy = (predictbox_xywh[:, :, 0:2] * defaultbox_wh * variances[:, 0:2]) + defaultbox_centerxy
        #   predictbox_xywh[:, :, 0:2] = (bbox_centerxy - defaultbox_centerxy) / defaultbox_wh / variances[:, 0:2]
        # solve: encoded_box[assign_mask, 2:4]
        #   (Take expression from decode_boxes)
        #   bbox_halfsize = defaultbox_wh * (np.exp(predictbox_xywh[:, :, 2:4] * variances[:, 2:4])) / 2
        #   np.exp(predictbox_xywh[:, :, 2:4] * variances[:, 2:4]) = bbox_halfsize / defaultbox_wh * 2
        #   predictbox_xywh[:, :, 2:4] * variances[:, 2:4] = np.log(bbox_halfsize / defaultbox_wh * 2)
        #   predictbox_xywh[:, :, 2:4] = np.log(bbox_size / defaultbox_wh) / variances[:, 2:4]
        encoded_box = np.zeros((len(self.priors), 4 + return_iou))
        if return_iou:
            encoded_box[assign_mask, -1] = iou[assign_mask]
        bbox_centerxy = (boundingbox[0:2] + boundingbox[2:4]) / 2
        bbox_size = boundingbox[2:4] - boundingbox[0:2]
        encoded_box[assign_mask, 0:2] = (bbox_centerxy - defaultbox_centerxy) / defaultbox_wh / variances[:, 0:2]
        encoded_box[assign_mask, 2:4] = np.log(bbox_size / defaultbox_wh) / variances[:, 2:4]

        return encoded_box

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.

        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.

        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        # assignment shape: (num_priors, 4: [x_min, y_min, x_max, y_max] + 1(class: background) + num_classes + 1(is_gt_box) + 7(all 0))
        assignment = np.zeros((len(self.priors), 4 + len(self.classes) + 1))
        # priorBox 全ての分類初期値は背景とする
        assignment[:, 4] = 1.0

        # encoded_boxes
        # shape: (boxes, num_priors, 4 + 1) 4: [x_min, y_min, x_max, y_max] 1: [iou]
        # rate: img size base
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, 0:4])
        # best_iou_value, best_iou_bxidx, zero_more_mask
        # shape: (num_priors,)
        # memo:
        #   best_iou_bxidx: Box idx to maximize iou of priorbox
        #   best_iou_value: Max iou of priorbox with each box
        best_iou_bxidx = encoded_boxes[:, :, 4].argmax(axis=0)
        best_iou_value = encoded_boxes[:, :, 4].max(axis=0)
        zero_more_mask = best_iou_value > 0
        # best_iou_bxidx:
        # memo: 各PriorsBox と 重複領域が最大の box の idx の配列から推論と正解の重複面積が0以上の要素を抽出
        best_iou_bxidx = best_iou_bxidx[zero_more_mask]
        encoded_boxes = encoded_boxes[:, zero_more_mask, :]
        # encoded_boxes[*]
        # shape: (priorbox_that_iou_zero_or_more, 4: [x_min, y_min, x_max, y_max])
        # memo:
        #   range(encoded_boxes.shape[1]) で box との座標オフセットを取得。
        #   この時、iou が最大のboxとの座標になるよう best_iou_bxidx で番地を切り替える
        assignment[zero_more_mask, 0:4] = encoded_boxes[best_iou_bxidx, range(encoded_boxes.shape[1]), 0:4]
        assignment[zero_more_mask, 4] = 0
        assignment[zero_more_mask, 5:-1] = boxes[best_iou_bxidx, 4:]
        assignment[zero_more_mask, -1] = 1
        return assignment

    def decode_boxes(self, predictions):
        """Convert bboxes from local predictions to shifted priors.

        # Arguments
            predictions: SSD300 output.

        # Return
            decode_bbox: Shifted priors.
        """
        # memo: 全座標・サイズの単位は比率
        # defaultbox_xyxy shape: (priorbox, 4: [min_x, min_y, max_x, max_y])
        defaultbox_xyxy = self.priors[:, 0:4]
        # defaultbox_centerxy shape: (priorbox, 4: [center_x, center_y])
        defaultbox_centerxy = (defaultbox_xyxy[:, 0:2] + defaultbox_xyxy[:, 2:4]) / 2
        # defaultbox_wh shape: (priorbox, 4: [width, height])
        defaultbox_wh = defaultbox_xyxy[:, 2:4] - defaultbox_xyxy[:, 0:2]
        variances = self.priors[:, 4:8]

        # predictbox_ltwh shape: (data, priorbox, 4: [x, y, width, height])
        predictbox_xywh = predictions[:, :, 0:4]
        # bbox_centerxy shape: (data, priorbox, 2: [x, y])
        bbox_centerxy = predictbox_xywh[:, :, 0:2] * defaultbox_wh * variances[:, 0:2]
        bbox_centerxy = bbox_centerxy + defaultbox_centerxy
        # bbox_halfsize shape: (data, priorbox, 2: [width, height])
        bbox_halfsize = np.exp(predictbox_xywh[:, :, 2:4] * variances[:, 2:4])
        bbox_halfsize = defaultbox_wh * bbox_halfsize / 2
        # boundingbox_ltrb shape: (data, priorbox, 4: [min_x, min_y, max_x, max_y])
        boundingbox_ltrb = np.concatenate((
            bbox_centerxy - bbox_halfsize,
            bbox_centerxy + bbox_halfsize), axis=2)
        boundingbox_ltrb = np.minimum(np.maximum(boundingbox_ltrb, 0.0), 1.0)

        return boundingbox_ltrb

    def detection_out(self, predictions, keep_top_k = 200, confidence_threshold_1 = 0.01, confidence_threshold_2 = 0.6):
        """Do non maximum suppression (nms) on prediction results.

        # Arguments
            predictions: Numpy array of predicted values.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step.
            confidence_threshold_1: Only consider detections,
                whose confidences are larger than a threshold.

        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        # define predictions(aka: prior_boxes_tensor)
        #   dim: [data, layer_width * layer_height * prior_box, 8:[xmin, ymin, xmax, ymax, *variances]]
        mbox_conf = predictions[:, :, 4:]
        mbox_conf_mask = mbox_conf > confidence_threshold_1
        decode_bbox = self.decode_boxes(predictions)

        for n in range(len(predictions)):
            results = list()
            for cid, class_label in enumerate(self.classes):
                # skip background class
                if cid == 0:
                    continue

                # 重複検出の除外
                mask_priorbox = mbox_conf_mask[n, :, cid]
                boxes_loc = decode_bbox[n, mask_priorbox]
                boxes_prb = mbox_conf[n, mask_priorbox, cid]
                indexes = tf.image.non_max_suppression(
                    boxes_loc, boxes_prb, self.top_k, iou_threshold=self.nms_thresh)

                # 結果の追加
                for i in indexes:
                    probability = boxes_prb[i]
                    location = boxes_loc[i]
                    if probability <= confidence_threshold_2:
                        continue
                    results.append(DetectedObject(cid, class_label, probability, *location))
            
            # スコア上位 keep_top_k 件を出力
            results.sort(key=lambda x: x.score, reverse=True)
            yield results[:keep_top_k]

    def show_pred(self, model: keras.Model, filenames: list, colors: list):
        inputs = list()
        images = list()
        for img_path in filenames:
            img = image.load_img(img_path)
            img = image.img_to_array(img)
            images.append(img)
            
            img = image.load_img(img_path, target_size=(300, 300))
            img = image.img_to_array(img)
            inputs.append(img)

        preds = imagenet_utils.preprocess_input(np.array(inputs))
        preds = model.predict(preds)
        preds = self.detection_out(preds)
        for item, img in zip(preds, images):
            plt.figure()
            self.show_detection(item, img, colors)

    def show_detection(self, detected_objs: list, img, colors: list):
        plt.imshow(img / 255)
        currentAxis = plt.gca()

        # show obj box in image
        for det_obj in detected_objs:
            score = det_obj.score
            label = det_obj.label
            color = colors[det_obj.label_id]
            
            # SSD300の出力する比率を実サイズへ変換
            xmin = int(round(det_obj.xmin * img.shape[1]))
            ymin = int(round(det_obj.ymin * img.shape[0]))
            xmax = int(round(det_obj.xmax * img.shape[1]))
            ymax = int(round(det_obj.ymax * img.shape[0]))
            width = xmax - xmin + 1
            height = ymax - ymin + 1
            
            # バウンディングボックスを表示
            currentAxis.add_patch(plt.Rectangle((xmin, ymin), width, height, edgecolor=color, fill=False))
            currentAxis.text(xmin, ymin, f"{score:0.2f}, {label}", bbox={'facecolor':color, 'alpha':0.5})

# class PredictResult:
@dataclasses.dataclass(frozen=True)
class DetectedObject:
    label_id: int
    label: str
    score: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float