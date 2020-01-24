import numpy as np
import typing as typ

from . import core
from . import data

def meanap(predicts: typ.List[data.DetectResult], grandtruths: typ.List[data.DetectResult], num_classes: int, iou_threshold: float = 0.5):
    nm_existed_classes = 0
    ap_existed_classes = 0
    for presitions, recalls in roccurve(predicts, grandtruths, num_classes, iou_threshold):
        # クラス毎のAPを求めて ap_existed_classes へ足す
        recall_prev = 0
        nm_per_class = 0
        for presition, recall in zip(presitions, recalls):
            nm_per_class = 1
            recall_diff = max((0, recall - recall_prev))
            recall_prev = recall
            ap_existed_classes += recall_diff * presition
        
        # 検知オブジェクトがあれば集計対象クラスとしてカウント
        # 後の map 計算用に使う。無ければカウントしない
        nm_existed_classes += nm_per_class

    return ap_existed_classes / nm_existed_classes

def roccurve(
    predicts: typ.List[data.DetectResult], grandtruths: typ.List[data.DetectResult],
    num_classes: int, iou_threshold: float = 0.5):

    # クラス毎の grand truth の数
    # recall 計算で分母になるので 0 以上になるようイプシロンを予め足しておく
    count_gt = np.full((num_classes), np.finfo(np.float).eps)

    # 検知されたオブジェクト数の分の 対応gtとのIOU, 確信度, 分類クラスを生成
    pred_iou_val_all: typ.List[float] = list()
    pred_cnf_val_all: typ.List[float] = list()
    pred_lbl_val_all: typ.List[float] = list()
    # 推論画像ごとに処理
    for objs_pred, objs_gt in zip(predicts, grandtruths):
        # label_id を取得
        pred_labels = np.array([item.label_id for item in objs_pred.boxes])
        gt_labels = np.array([item.label_id for item in objs_gt.boxes])

        # クラス毎の正解オブジェクト数を集計。recallの計算時の分母
        for label_id in gt_labels:
            count_gt[label_id] += 1
        
        # iou を計算 step.1 (pred_iou_val: ndarray[pred, gt] -> float)
        gt_boxes = np.array([(item.xmin, item.ymin, item.xmax, item.ymax) for item in objs_gt.boxes])
        pred_boxes = np.array([(item.xmin, item.ymin, item.xmax, item.ymax) for item in objs_pred.boxes])
        if pred_boxes.size == 0 or gt_boxes.size == 0:
            continue
        pred_iou_val = iou(pred_boxes, gt_boxes)
        
        # iou を計算 step.2 (pred_iou_val: ndarray[pred] -> float)
        # 異なるクラスの box の iou は 0 にするため、マスクを適用
        pred_gt_crossmask = (pred_labels[:, np.newaxis] == gt_labels[np.newaxis, :])
        pred_gt_crossmask = pred_gt_crossmask.astype(np.int)
        pred_iou_val = pred_iou_val * pred_gt_crossmask
        # 各推論に最もマッチした正解値との iou のみを出す
        pred_iou_idx = np.argmax(pred_iou_val, axis=1)
        pred_iou_val = [item[idx] for item, idx in zip(pred_iou_val, pred_iou_idx)]

        pref_scr_val = np.array([item.score for item in objs_pred.boxes])
        pred_iou_val_all.extend(pred_iou_val)
        pred_cnf_val_all.extend(pred_iou_val * pref_scr_val)
        pred_lbl_val_all.extend(pred_labels.tolist())

    # 適合率、再現率を求める
    presition = [list() for i in range(num_classes)]
    recall = [list() for i in range(num_classes)]
    count_tp = np.zeros((num_classes))
    count_pred = np.zeros((num_classes))
    # 確信度が高い順に処理する
    pred_cnf_srt = np.argsort(pred_cnf_val_all, axis=0)
    pred_cnf_srt = pred_cnf_srt[::-1]
    for pred_idx in pred_cnf_srt:
        pred_lbl = pred_lbl_val_all[pred_idx]
        # True Positive (iou > 0.5) を集計
        count_tp[pred_lbl] += int(pred_iou_val_all[pred_idx] > iou_threshold)
        # True Positive + False Positive (全ての検知数) を集計
        count_pred[pred_lbl] += 1
        # クラス毎に適合率、再現率を求める
        presition[pred_lbl].append(count_tp[pred_lbl] / count_pred[pred_lbl])
        recall[pred_lbl].append(count_tp[pred_lbl] / count_gt[pred_lbl])

    # グラフにプロットした際に線が下に付くようにする
    for pred_idx in pred_cnf_srt:
        pred_lbl = pred_lbl_val_all[pred_idx]
        presition[pred_lbl].append(0.0)
        recall[pred_lbl].append(recall[pred_lbl][-1])

    return zip(presition, recall)

def iou(boxes_a: np.ndarray, boxes_b: np.ndarray):
    """
    Arguments
    --------------------
    boxes_a: ndarray[box, 4:(minx, miny, maxx, maxy)] -> float
    boxes_b: ndarray[box, 4:(minx, miny, maxx, maxy)] -> float

    Return
    --------------------
    iou: ndarray[box_a, box_b] -> float (from 0 to 1)
    """
    iou = np.empty((boxes_a.shape[0], boxes_b.shape[0]))
    boxes_sz_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    boxes_sz_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    for i, box_a in enumerate(boxes_a):
        # compute intersection
        inter_xymin = np.maximum(box_a[0:2], boxes_b[:, 0:2])
        inter_xymax = np.minimum(box_a[2:4], boxes_b[:, 2:4])
        inter_wh = np.maximum(inter_xymax - inter_xymin, 0)
        inter_sz = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        union = boxes_sz_a[i] + boxes_sz_b - inter_sz
        # compute iou
        iou[i] = inter_sz / union

    return iou