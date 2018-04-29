import glob
import json

import numpy as np


def evaluate_bounding_boxes(annotation_boxes):
    predicted_boxes = []
    json_list = glob.glob('FaceDataset/validation/images/out/*.json')
    for json_file in json_list:
        data = json.load(open(json_file))
        name = json_file[34:]
        for obj in data:
            confidence = int(obj['confidence'])
            xmin = int(obj['topleft']['x'])
            ymin = int(obj['topleft']['y'])
            xmax = int(obj['bottomright']['x'])
            ymax = int(obj['bottomright']['x'])
            predicted_boxes.append(EvalBoundBox(name, confidence, xmin, ymin, xmax, ymax))

    iou = 0
    n = 0
    confidence = 0
    for predicted_box in predicted_boxes:
        confidence += predicted_box.confidence
        for true_box in annotation_boxes:
            if true_box.name[:-4] == predicted_box.name[:-5]:
                temp_iou = box_iou(true_box, predicted_box)
                if temp_iou > 0.1:
                    iou += temp_iou
                    n += 1
    if n != 0:
        return iou / n, confidence / n
    else:
        return 0, 0


class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))


class EvalBoundBox:
    def __init__(self, confidence, name, xmin, ymin, xmax, ymax):
        self.name = name
        self.confidence = confidence
        self.x = xmin
        self.y = ymin
        self.w = xmax - xmin
        self.h = ymax - ymin


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;


def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;


def box_iou(a, b):
    if box_union(a, b) == 0:
        return 0
    else:
        return box_intersection(a, b) / box_union(a, b);


def prob_compare(box):
    return box.probs[box.class_num]


def prob_compare2(boxa, boxb):
    if (boxa.pi < boxb.pi):
        return 1
    elif (boxa.pi == boxb.pi):
        return 0
    else:
        return -1
