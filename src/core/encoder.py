import numpy as np
import tensorflow as tf
import typing as typ
from tensorflow.keras.preprocessing import image

from .. import metrics
from .. import data

class BoxEncoder:
    def __init__(self, priors: np.ndarray, num_classes: int, overlap_threshold: float = 0.5, nms_thresh: float = 0.45, top_k: int = 400):
        """Utility class to do some stuff with bounding boxes and priors.

        Arguments
        ----------------------
        priors: Priors and variances, numpy tensor of shape (num_priorboxes, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold.
        top_k: Number of total bboxes to be kept per image after nms step.

        References
        ----------------------
        https://arxiv.org/abs/1512.02325

        """
        priors = priors.astype(np.float32)
        self._priorboxes = priors
        self._overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k

        self.encoded_shape = tf.TensorShape([len(priors), 6 + num_classes])
        self.decoded_shape = tf.TensorShape([None, 4 + num_classes])

        # memo: 全座標・サイズの単位は比率
        self._priorboxes_centerxy = (priors[:, 0:2] + priors[:, 2:4]) / 2
        self._priorboxes_wh = priors[:, 2:4] - priors[:, 0:2]
        self._variances = priors[:, 4:8]

    def encode(self, bboxes: typ.List[np.ndarray]) -> np.ndarray:
        """Encode decoded_boxes to priors for training.

        Arguments
        --------------------
        decoded_boxes: Box, numpy tensor of shape (num_boxes, 4 + classes),
        num_classes within background.

        Return
        --------------------
        encoded_boxes: Tensor with encoded decoded_boxes,
        numpy tensor of shape (num_boxes, priors, 4 + 1:[background] + classes + 1:[hard_nega_mining]),
        """
        encoded_boxes_all = list()
        for decoded_boxes in bboxes:
            # 計算範囲のマスクを作成
            # assign_mask (priorbox と box の対応表) を生成
            #   shape: (decoded_boxes, priors)
            # memo:
            #   _priorboxes: shape: (priorbox, 8: [x_min, y_min, x_max, y_max, var1, var2, var3, var4])
            #   overlap_mask: priorbox 毎に 閾値を超える iou が出る box を求める
            #   topiou_mask: box 毎に 最大 iou が出る priorbox を求める
            pbox_num, _ = self._priorboxes.shape
            bbox_num, feature_num = decoded_boxes.shape
            # 物体が無い場合は全て背景扱いの出力を生成 (後続処理を続行すると np.argmax で止まるので分岐)
            if (bbox_num == 0):
                output = np.zeros((1, pbox_num, feature_num + 2))
                output[:, :, 4] = 1
                encoded_boxes_all.append(output)
                continue
            
            class_num = feature_num - 4
            iou_cross = metrics.iou(decoded_boxes[:, 0:4], self._priorboxes[:, 0:4])
            overlap_mask = iou_cross > self._overlap_threshold
            overlap_mask[range(bbox_num), np.argmax(iou_cross, axis=1)] = True
            topiou_box_idx = np.argmax(iou_cross, axis=0)
            topiou_mask = np.full_like(iou_cross, False).astype(np.bool)
            topiou_mask[topiou_box_idx, range(pbox_num)] = True
            assign_mask = overlap_mask & topiou_mask

            # 座標計算用の準備
            # define
            #   bbox_centerxy   shape: (bbox, 1:[priorbox], 2:[x, y])
            #   bbox_wh         shape: (bbox, 1:[priorbox], 2:[x, y])
            #   variances       shape: (bbox, priorbox, 4:(xmin, ymin, xmin, ymax))
            #   pboxes_centerxy shape: (bbox, priorbox, 2:[x, y])
            #   pboxes_wh       shape: (bbox, priorbox, 2:[x, y])
            # memo:
            #   bbox_mask の存在意義についてメモ. bbox が重なり過ぎる場合に
            #   topiou_mask 部分 で box の欠落があり得る. その際の配列サイズズレ対策が bbox_mask
            # : = np.any(assign_mask, axis=1)
            bbox_centerxy = (decoded_boxes[:, 0:2] + decoded_boxes[:, 2:4]) / 2
            bbox_centerxy = bbox_centerxy[:, np.newaxis, :]
            bbox_wh = decoded_boxes[:, 2:4] - decoded_boxes[:, 0:2]
            bbox_wh = bbox_wh[:, np.newaxis, :]
            pboxes_centerxy = np.broadcast_to(self._priorboxes_centerxy, (bbox_num, pbox_num, 2))
            pboxes_wh = np.broadcast_to(self._priorboxes_wh, (bbox_num, pbox_num, 2))
            variances = np.broadcast_to(self._variances, (bbox_num, pbox_num, 4))

            # 座標計算用の本番
            # encoded_boxes
            #   shape: (priorboxes, 4:[x_min, y_min, x_max, y_max] + 1:[hard_nega_mining])
            #   solve: encoded_boxes[assign_mask, 0:2]
            #     (Take expression from decode)
            #     bbox_centerxy = (encoded_boxes[:, :, 0:2] * self._priorboxes_wh * self._variances[:, 0:2]) + pboxes_centerxy
            #     encoded_boxes[:, :, 0:2] = (bbox_centerxy - pboxes_centerxy) / self._priorboxes_wh / self._variances[:, 0:2]
            #   solve: encoded_boxes[assign_mask, 2:4]
            #     (Take expression from decode)
            #     bbox_halfwh = self._priorboxes_wh * (np.exp(encoded_boxes[:, :, 2:4] * self._variances[:, 2:4])) / 2
            #     np.exp(encoded_boxes[:, :, 2:4] * self._variances[:, 2:4]) = bbox_halfwh / self._priorboxes_wh * 2
            #     encoded_boxes[:, :, 2:4] * self._variances[:, 2:4] = np.log(bbox_halfwh / self._priorboxes_wh * 2)
            #     encoded_boxes[:, :, 2:4] = np.log(bbox_size / self._priorboxes_wh) / self._variances[:, 2:4]
            encoded_xy = (bbox_centerxy - pboxes_centerxy) / pboxes_wh / variances[:, :, 0:2]
            encoded_wh = np.log(bbox_wh / pboxes_wh) / variances[:, :, 2:4]
            encoded_conf = decoded_boxes[:, np.newaxis, 4:]
            encoded_conf = np.broadcast_to(encoded_conf, (bbox_num, pbox_num, class_num))
            encoded_zero = np.zeros((bbox_num, pbox_num, 1))
            # encoded_boxes shape: (bbox, priorbox, 4 + 1:[background] + class_num + 1:[hard_nega_mining])
            encoded_boxes = np.concatenate((encoded_xy, encoded_wh, encoded_zero, encoded_conf, encoded_zero), axis=2)
            encoded_boxes = encoded_boxes * assign_mask.astype(np.int)[:, :, np.newaxis]
            encoded_boxes = np.sum(encoded_boxes, axis=0)

            detect_mask = np.any(assign_mask, axis=0)
            background_mask = ~detect_mask
            # 物体判定のボックスの 物体としてマーク (hard_nega_mining) する
            encoded_boxes[detect_mask, -1] = 1
            # 背景判定のボックスのクラス分類を背景に指定
            encoded_boxes[background_mask, 4] = 1
            encoded_boxes_all.append(encoded_boxes[np.newaxis])
        
        encoded_boxes_all = np.vstack(encoded_boxes_all)
        return encoded_boxes_all

    def decode(self, encoded_boxes: np.ndarray, keep_top_k=200, conf_threshold_1=0.01, conf_threshold_2=0.6) -> typ.List[np.ndarray]:
        """Do non maximum suppression (nms) on prediction results.

        Arguments
        --------------------
        encoded_boxes:    Numpy array of predicted values.
        keep_top_k:       Number of total bboxes to be kept per image after nms step.
        conf_threshold_1: Only consider detections, whose confidences are larger than a threshold.

        Return
        --------------------
        results: List of encoded_boxes for every picture.
                 Each prediction is: [label, confidence, xmin, ymin, xmax, ymax]
        """
        # define:
        #   shifted_pboxes shape: (data, priorbox, 4: [x, y, width, height] + 1:[background] + classes + 1:[hard_nega_mining])
        # source:
        #   encoded_boxes        shape: (data, priorbox, 4: [x, y, width, height] + 1:[background] + classes + 1:[hard_nega_mining])
        #   _priorboxes_wh       shape: (priorbox, 2: [x, y])
        #   _priorboxes_centerxy shape: (priorbox, 2: [x, y])
        #   _variances           shape: (priorbox, 4: [x, y, w, h])
        encoded_boxes = encoded_boxes.astype(np.float32)
        bbox_centerxy = encoded_boxes[:, :, 0:2] * self._priorboxes_wh * self._variances[:, 0:2]
        bbox_centerxy = bbox_centerxy + self._priorboxes_centerxy
        bbox_halfwh = np.exp(encoded_boxes[:, :, 2:4] * self._variances[:, 2:4])
        bbox_halfwh = self._priorboxes_wh * bbox_halfwh / 2
        shifted_pboxes = np.concatenate((
            np.maximum(np.minimum(bbox_centerxy - bbox_halfwh, 1.0), 0.0),
            np.maximum(np.minimum(bbox_centerxy + bbox_halfwh, 1.0), 0.0), encoded_boxes[:, :, 4:]), axis=2)
        _, _, feature_num = shifted_pboxes.shape
        class_withbk_num = feature_num - 4 - 1

        bboxes = list()
        for n in range(len(shifted_pboxes)):
            # define:
            #   decoded_boxes shape: (data, box, 4: [x, y, width, height] + 1:[background] + classes)
            #   class_withbk_num shape (feature_num - 4:[xmin, ymin, xmax, ymax] - 1:[hard_nega_mining])
            decoded_boxes: typ.List[np.ndarray] = list()
            # start from 1: skip background class
            for class_id in range(1, class_withbk_num):
                # 重複検出の除外
                over_threshold_mask = shifted_pboxes[n, :, 4 + class_id] > conf_threshold_1
                boxes_loc = shifted_pboxes[n, over_threshold_mask, 0:4]
                boxes_cnf = shifted_pboxes[n, over_threshold_mask, 4 + class_id]
                indexes = tf.image.non_max_suppression(boxes_loc, boxes_cnf, self._top_k, iou_threshold=self._nms_thresh)
                indexes = indexes.numpy()

                filtered_pboxes = shifted_pboxes[n, over_threshold_mask][indexes]
                filtered_pboxes_mask = filtered_pboxes[:, 4 + class_id] > conf_threshold_2
                filtered_pboxes = filtered_pboxes[filtered_pboxes_mask]
                # 特徴量から背景とハードネガマイニングの2次元を削除
                withoutbk_pboxes_mask = np.arange(feature_num) != 4
                withoutbk_pboxes_mask[-1] = False
                filtered_pboxes = filtered_pboxes[:, withoutbk_pboxes_mask]
                decoded_boxes.append(filtered_pboxes)

            # スコア上位 keep_top_k 件を出力
            decoded_boxes = np.vstack(decoded_boxes)
            decoded_boxes_cnf = decoded_boxes[:, 4:feature_num-1]
            decoded_boxes_idx = list(range(len(decoded_boxes)))
            cnf_top_class_idx = np.argmax(decoded_boxes_cnf, axis=1)
            decoded_boxes_cnf = decoded_boxes_cnf[decoded_boxes_idx, cnf_top_class_idx]
            dat_sort_idx = np.argsort(decoded_boxes_cnf)
            dat_sort_idx = dat_sort_idx[::-1]

            decoded_boxes = decoded_boxes[dat_sort_idx]
            decoded_boxes = decoded_boxes[:keep_top_k]
            bboxes.append(decoded_boxes)
    
        return bboxes
