import numpy as np
import src.metrics as metrics
import src.data as data

def test_iou():
    # boxes: [box, 4:(minx, miny, maxx, maxy)]
    input_boxes = np.array([
        [0, 0, 1, 1],
        [2, 1, 6, 4],
        [3, 2, 4, 3],
        [5, 2, 7, 5],
    ])
    # val_expect: [box_a, box_b]
    val_expect = np.array([
        [1, 0, 0, 0],
        [0, 1, 1/12, 2/16],
        [0, 1/12, 1, 0],
        [0, 2/16, 0, 1],
    ])
    val_actual = metrics.iou(input_boxes, input_boxes)
    val_err = np.sum(np.abs(val_expect - val_actual))
    assert val_err == 0

def test_roc():
    input_preds = [
        # case. 1 検知オブジェクトがある場合
        data.DetectResult([
            # case. 1.1 (最大 iou が選ばれること)
            data.DetectedObject(0, "foo1", 1.0, 0, 0, 3, 1),
            # case. 1.2 (確信度 cnf が列挙順に反映されること)
            data.DetectedObject(0, "foo1", 1.0, 0, 1, 3, 2),
            # case. 1.3 (クラス間は独立してること)
            data.DetectedObject(1, "foo2", 1.0, 0, 0, 3, 1),
        ]),
        # case. 2 検知オブジェクトがある場合
        data.DetectResult([]),
    ]
    input_gts = [
        data.DetectResult([
            # case. 1
            data.DetectedObject(0, "foo1", 1.0, 0, 0, 2, 1),
            data.DetectedObject(0, "foo1", 1.0, 2, 0, 3, 1),
            # case. 2
            data.DetectedObject(0, "foo1", 0.1, 0, 1, 3, 2),
            data.DetectedObject(0, "foo1", 1.0, 0, 1, 2, 2),
            # case. 3
            data.DetectedObject(1, "foo2", 1.0, 0, 0, 3, 1),
        ]),
        # case. 2
        data.DetectResult([]),
    ]
    val_actual = list(metrics.roc(input_preds, input_gts, 3))
    # 結果検証のコードはそのうち... (ヽ´ω`)