import glob
import json
import xml.etree.ElementTree as ET

import numpy as np


def evaluate_bounding_boxes():
    predicted_boxes = []
    json_list = glob.glob('darkflow/FaceDataset/images/out/*.json')
    for json_file in json_list:
        data = json.load(json_file)
        for obj in data:
            xmin = int(obj['topleft']['x'])
            ymin = int(obj['topleft']['y'])
            xmax = int(obj['bottomright']['x'])
            ymax = int(obj['bottomright']['x'])
            predicted_boxes.append(EvalBoundBox(xmin, ymin, xmax, ymax))

    annotation_boxes = []
    xml_list = glob.glob('darkflow/FaceDataset/annotations/*.xml')
    for xml_file in xml_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            annotation_boxes.append(EvalBoundBox(xmin, ymin, xmax, ymax))

    iou = 0
    n = 0
    for true_box in annotation_boxes:
        for predicted_box in predicted_boxes:
            temp_iou = box_iou(true_box, predicted_box)
            if iou > 0:
                iou += iou
                n += 1
    return iou/n


class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.class_num = classes
        self.probs = np.zeros((classes,))


class EvalBoundBox:
    def __init__(self, xmin, ymin, xmax, ymax):
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
