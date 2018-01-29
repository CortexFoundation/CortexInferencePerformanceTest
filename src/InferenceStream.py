import cv2
import mxnet as mx
import numpy as np
import datetime
import traceback
from multiprocessing import Process,Queue
from collections import namedtuple
import json
import time
import os
import random
import Inference
class InferenceStream(Inference.Inference):
    def __init__(self,config):
        Inference.Inference.__init__(self,config)
        self.gpu_num = config["gpu_num"]
        self.Batch = namedtuple('Batch', ['data'])

    def predict(self,imgs,mod):
        mod.forward(self.Batch([mx.nd.array(imgs)]))
        return None

    def predictFromQueue(self,inputQueue,outputQueue,gpu_id):
        gpu_model_list = []
        for model_name in self.model_names:
            mod,load_model_time = self.loadModel(os.path.join(self.model_dir,model_name),type="gpu",gpu_id = gpu_id)
            gpu_model_list.append(mod)
        outputQueue.put(None)
        print ("Load Model Complete, GPU id %d"%gpu_id)
        while True:
            imgs,model_id = inputQueue.get()
            time = self.predict(imgs,gpu_model_list[model_id])
            outputQueue.put(time)

    def rmInvalidData(self,rm_img = False):
        if rm_img:
            predict_files = []
            for filename in os.listdir(self.img_dir):
                if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
                    predict_files.append(filename)
            for name in predict_files:
                try:
                    img = self.getImage(name)
                except Exception,ex:
                    os.remove(os.path.join(self.img_dir,name))
        self.predict_files = []
        self.img_data = {}
        for filename in os.listdir(self.img_dir):
            if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
                self.predict_files.append(filename)
                self.img_data[filename] = self.getImage(filename)

    def testInference(self,times = 10000):
        inputQueue = Queue()
        outputQueue = Queue()
        processPool = []
        for i in range(self.gpu_num):
            processPool.append(Process(target=self.predictFromQueue,args=(inputQueue,outputQueue,i,)))
        for i in range(self.gpu_num):
            processPool[i].daemon = True
            processPool[i].start()
        self.rmInvalidData()
        for i in range(self.gpu_num):
            outputQueue.get()
        cpu_infer_time = []
        gpu_infer_time = []
        begin = time.time()
        for i in range(times):
            sampled = random.sample(self.predict_files,self.batchsize)
            imgs = []
            for name in sampled:
                imgs.append(self.img_data[name])
            rand_model = random.randint(0,self.gpu_num-1)
            inputQueue.put((imgs,rand_model))
        for i in range(times):
            t = outputQueue.get()
        end = time.time()
        print "Inference Total Time Used:", end-begin
