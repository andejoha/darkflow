from .defaults import argHandler #Import the default arguments
import os
from .net.build import TFNet
import matplotlib.pyplot as plt

def cliHandler(args):
    global loss_hist = []
    global iou_hist = []

    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    
    requiredDirectories = [FLAGS.imgdir, FLAGS.binary, FLAGS.backup, os.path.join(FLAGS.imgdir,'out')]
    if FLAGS.summary:
        requiredDirectories.append(FLAGS.summary)

    _get_dir(requiredDirectories)

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    tfnet = TFNet(FLAGS)
    
    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb: 
            exit('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')

    tfnet.predict()

    plt.subplot(211)
    plt.plot(loss_hist)
    plt.ylabel('Loss')

    plt.subplot(212)
    plt.plot(iou_hist)
    plt.ylabel('Intersection over Union')
    plt.xlabel('Steps')
    plt.title('Loss & IoU')

    plt.show()
