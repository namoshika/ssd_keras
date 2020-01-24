import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import typing as typ
from pathlib import Path
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image

from . import data
from . import metrics
from . import core

class Utility:
    def __init__(self, encoder: core.BoxEncoder, classes: typ.List[str], grandtruth, images_size) -> None:
        self.classes = classes
        self.colors = plt.cm.hsv(np.linspace(0, 1, len(self.classes))).tolist()
        self.encoder = encoder
        self.grandtruth = grandtruth
        self.images_size = images_size

    def get_predict(self, model: keras.Model, filenames: typ.List[str], voc_dir_path: str, batch_size = 500) -> typ.List[data.DetectResult]:
        files = [Path(voc_dir_path) / "JPEGImages" / name for name in filenames]
        predicts_all = list()
        for i in range(0, len(files), batch_size):
            files_subset = files[i:i+batch_size]
            images_shape = map(lambda filepath: image.load_img(filepath), files_subset)
            images_shape = map(lambda img: image.img_to_array(img).shape, images_shape)
            inputs = map(lambda filepath: image.load_img(filepath, target_size=(300, 300)), files_subset)
            inputs = map(lambda img: image.img_to_array(img), inputs)
            inputs = map(lambda ary: imagenet_utils.preprocess_input(ary), inputs)
            inputs = np.array(list(inputs))

            predicts = model.predict(inputs)
            predicts = self.encoder.decode(predicts)
            predicts = [data.DetectResult.from_decoded_array(bboxes, self.classes, img_shape) for bboxes, img_shape in zip(predicts, images_shape)]
            predicts_all.extend(predicts)
        
        return predicts_all

    def get_grandtruth(self, filenames: typ.List[str]) -> typ.List[data.DetectResult]:
        grandtruths = [(self.grandtruth[name], self.images_size[name]) for name in filenames]
        grandtruths = [data.DetectResult.from_decoded_array(bboxes, self.classes, img_size) for bboxes, img_size in grandtruths]
        return grandtruths

    def show_predict_with_gt(self, model: keras.Model, filenames: typ.List[str], voc_dir_path: str) -> None:
        # 表示用画像を生成
        filepaths = [Path(voc_dir_path) / "JPEGImages" / name for name in filenames]
        images = map(lambda filepath: image.load_img(filepath), filepaths)
        images = map(lambda img: image.img_to_array(img), images)

        # 推論と正解を生成
        grandtruths = self.get_grandtruth(filenames)
        predicts = self.get_predict(model, filenames, voc_dir_path)

        # iterate images
        for img, grd, prd in zip(images, grandtruths, predicts):
            plt.figure(figsize=(12, 18))
            # predict
            plt.subplot(1, 2, 1)
            plt.title("predict")
            self.show_detection(prd, img)
            # grandtruth
            plt.subplot(1, 2, 2)
            plt.title("grandtruth")
            self.show_detection(grd, img)

    def show_predict(self, model: keras.Model, filenames: typ.List[str]) -> None:
        images = map(lambda filepath: image.load_img(filepath), filenames)
        images = map(lambda img: image.img_to_array(img), images)
        images = list(images)
        inputs = map(lambda filepath: image.load_img(filepath, target_size=(300, 300)), filenames)
        inputs = map(lambda img: image.img_to_array(img), inputs)
        inputs = map(lambda ary: imagenet_utils.preprocess_input(ary), inputs)
        inputs = np.array(list(inputs))
        images_size = [item.shape[0:2] for item in images]

        preds = model.predict(inputs)
        preds = self.encoder.decode(preds)
        for item, img, sz in zip(preds, images, images_size):
            detres = data.DetectResult.from_decoded_array(item, self.classes, sz)
            plt.figure()
            self.show_detection(detres, img)

    def show_roccurve(self, predicts: typ.List[data.DetectResult], grandtruths: typ.List[data.DetectResult], iou_threshold: float = 0.5) -> None:
        plt.figure(figsize=(18, 8), facecolor="white")
        plt.subplots_adjust(wspace=0.5, hspace=0.8)
        num_classes = len(self.classes)
        for i, (presitions, recalls) in enumerate(metrics.roccurve(predicts, grandtruths, num_classes, iou_threshold)):
            # col:6, row:colで割った結果を整数に切り上げ
            col = 6
            row = -(-num_classes // col)
            # プロット
            plt.subplot(row, col, i + 1)
            plt.title(self.classes[i])
            plt.xlabel("recall")
            plt.xlim(0, 1.1)
            plt.ylabel("presition")
            plt.ylim(0, 1.1)
            plt.plot(recalls, presitions)

    def show_detection(self, detecte_result: data.DetectResult, img: np.ndarray) -> None:
        plt.imshow(img / 255)
        currentAxis = plt.gca()

        # show obj box in image
        for det_obj in detecte_result.boxes:
            score = det_obj.score
            label = det_obj.label
            color = self.colors[det_obj.label_id]
            
            # SSD300の出力する比率を実サイズへ変換
            xmin = int(round(det_obj.xmin))
            ymin = int(round(det_obj.ymin))
            xmax = int(round(det_obj.xmax))
            ymax = int(round(det_obj.ymax))
            width = xmax - xmin + 1
            height = ymax - ymin + 1
            
            # バウンディングボックスを表示
            currentAxis.add_patch(plt.Rectangle((xmin, ymin), width, height, edgecolor=color, fill=False))
            currentAxis.text(xmin, ymin, f"{score:0.2f}, {label}", bbox={'facecolor':color, 'alpha':0.5})

    def evaluate_meanap(self, predicts: typ.List[data.DetectResult], grandtruths: typ.List[data.DetectResult], iou_threshold: float = 0.5) -> float:
        res = metrics.meanap(predicts, grandtruths, len(self.classes), iou_threshold)
        return res
