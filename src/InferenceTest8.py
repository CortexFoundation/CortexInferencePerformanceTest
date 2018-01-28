import cv2
import mxnet as mx
import numpy as np
import datetime
import traceback
from multiprocessing import Process,Queue
# define a simple data batch
from collections import namedtuple
import json
import time
import os
import random
gpu_num = 8
batchsize = 1
Batch = namedtuple('Batch', ['data'])
model_names = ["caffenet","Inception-BN","nin","resnet-152","RN101-5k500","squeezenet_v1.1","vgg16","vgg19"]
img_data = {}

def get_image(name, show=False):
    # download and show the image
    img = cv2.cvtColor(cv2.imread("../data/"+name), cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    # img = img[np.newaxis, :]
    return img

def load_model(name,id = 0):
    begin = time.time()
    sym, arg_params, aux_params = mx.model.load_checkpoint(name, 0)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(id), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (batchsize, 3, 224, 224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    end = time.time()
    return mod,end-begin

def predict(imgs,mod):
    mod.forward(Batch([mx.nd.array(imgs)]))
    return None

def get_FileSize(filePath):
    filePath = unicode(filePath,'utf8')
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024*1024)
    return round(fsize,2)

def predictFromQueue(inputQueue,outputQueue,gpu_id):
    gpu_model_list = []
    for model_name in model_names:
        mod,load_model_time = load_model(path_dir + model_name,id = gpu_id)
        gpu_model_list.append(mod)
    outputQueue.put(None)
    print ("Load Model Complete, GPU id %d"%gpu_id)
    while True:
        imgs,model_id = inputQueue.get()
        time = predict(imgs,gpu_model_list[model_id])
        outputQueue.put(time)

if __name__ == "__main__":
    path_dir = "../model/"
    inputQueue = Queue()
    outputQueue = Queue()
    processPool = []
    for i in range(gpu_num):
        processPool.append(Process(target=predictFromQueue,args=(inputQueue,outputQueue,i,)))
    for i in range(gpu_num):
        processPool[i].daemon = True
        processPool[i].start()

    predict_files = []
    for filename in os.listdir('../data'):
        if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
            predict_files.append(filename)
            img_data[filename] = get_image(filename,show=False)
    # print len(predict_files)
    # for name in predict_files:
    #     try:
    #         img = get_image(name)
    #         print img.shape
    #     except Exception,ex:
    #         os.remove("../data/"+name)
    #         print name
    # exit()
    for i in range(gpu_num):
        outputQueue.get()
    times = 100000
    cpu_infer_time = []
    gpu_infer_time = []
    begin = time.time()
    for i in range(times):
        sampled = random.sample(predict_files,batchsize)
        imgs = []
        for name in sampled:
            imgs.append(img_data[name])
        rand_model = random.randint(0,gpu_num-1)
        inputQueue.put((imgs,rand_model))
    for i in range(times):
        t = outputQueue.get()
    end = time.time()
    print "Inference Total Time Used:", end-begin
