import os

import matplotlib.pyplot as plt

from .defaults import argHandler  # Import the default arguments
from .net.build import TFNet


def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)

    requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir, 'out')]
    if FLAGS.summary:
        requiredDirectories.append(FLAGS.summary)

    _get_dir(requiredDirectories)

    # fix FLAGS.load to appropriate type
    try:
        FLAGS.load = int(FLAGS.load)
    except:
        pass

    tfnet = TFNet(FLAGS)

    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        print('Enter training ...')
        loss_hist, iou_hist, avg_confidence_hist, steps = tfnet.train()
        print('loss_hist =', loss_hist)
        print('iou_hist =', iou_hist)

        if not os.path.exists('plot/'):
            os.makedirs('plot')
        plt.subplot(311)
        plt.title('Loss, IoU & Confidence')
        plt.plot(loss_hist)
        plt.ylabel('Loss')
        plt.subplot(312)
        plt.plot(iou_hist)
        plt.ylabel('Intersection over Union')
        plt.subplot(313)
        plt.plot(avg_confidence_hist)
        plt.ylabel('Confidence')
        plt.xlabel('Steps')
        plt.savefig('plot/Loss_&_IoU.png')

        if not FLAGS.savepb:
            exit('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb();
        exit('Done')

    tfnet.predict()
