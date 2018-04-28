import os
import pickle
import time
from multiprocessing.pool import ThreadPool
import xml.etree.ElementTree as ET
import numpy as np
import glob
from .. import cli
from ..utils.box import evaluate_bounding_boxes, EvalBoundBox
import matplotlib.pyplot as plt

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()


def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt:
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)

    annotation_boxes = []
    xml_list = glob.glob('FaceDataset/validation/annotations/*.xml')
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

    args = ['flow', '--model', 'cfg/tiny-yolo-voc-face.cfg', '--load', '-1', '--imgdir',
            'FaceDataset/validation/images', '--json', '--gpu', '1.0']
    cli.cliHandler(args)
    iou = evaluate_bounding_boxes(annotation_boxes)
    print('Intersection over Union:', iou)
    return iou


def train(self):
    iou_hist = []
    loss_hist = []
    steps = []

    loss_ph = self.framework.placeholders
    loss_mva = None;
    profile = list()

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    for i, (x_batch, datum) in enumerate(batches):
        print('lost_hits =', loss_hist)
        print('iou_hist =', iou_hist)
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key]
            for key in loss_ph}
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = self.FLAGS.load + i + 1

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        loss_hist.append(loss)
        steps.append(step_now)
        profile += [(loss, loss_mva)]

        ckpt = (i + 1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt:
            iou_hist.append(_save_ckpt(self, *args))
        plt.figure(step_now)
        plt.subplot(211)
        plt.plot(loss_hist)
        plt.ylabel('Loss')
        plt.subplot(212)
        plt.plot(iou_hist)
        plt.ylabel('Intersection over Union')
        plt.xlabel('Steps')
        plt.title('Loss & IoU')
        plt.show()

    if ckpt:
        iou_hist.append(_save_ckpt(self, *args))

    return loss_hist, iou_hist


def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
        'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp: this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo


import math


def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp: np.concatenate(inp_feed, 0)}
        # self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time();
        last = stop - start
        # self.say('Total time = {}s / {} inps = {} ips'.format(
        last, len(inp_feed), len(inp_feed) / last

        # Post processing
        # self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
        self.framework.postprocess(
            prediction, os.path.join(inp_path, this_batch[i])))(*p),
                                                               enumerate(out))
        stop = time.time();
        last = stop - start

        # Timing
        # self.say('Total time = {}s / {} inps = {} ips'.format(
        last, len(inp_feed), len(inp_feed) / last
