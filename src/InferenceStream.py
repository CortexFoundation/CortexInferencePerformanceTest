import cv2
import mxnet as mx
import numpy as np
import datetime
import traceback
# define a simple data batch
from collections import namedtuple
import json
import time
import os
import random
import Inference
class InferenceStream(Inference.Inference):
    def __init__(self,config):
        super(config)
        self.gpu_num = config["gpu_num"]

    def predict(self,names,mod):
        mod.forward(Batch([mx.nd.array(imgs)]))
        return None

    def rmInvalidData(self,files_name,rm_img = False):
        if rm_img:
        for name in files_name:
            try:
                img = self.getImage(name)
            except Exception,ex:
                os.remove(os.path.join(self.img_dir,name))
        self.predict_files = []
        for filename in os.listdir(self.img_dir):
            if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
                self.predict_files.append(filename)

    def loadModel(self):
        self.cpu_model_list = []
        self.gpu_model_list = []
        self.cpu_model_time = []
        self.gpu_model_time = []
        for model_name in self.model_names:
            mod,load_model_time = self.loadModel(os.path.join(self.model_dir, model_name),type="cpu")
            self.cpu_model_list.append(mod)
            self.cpu_model_time.append(
                {
                    "model_name":model_name,
                    "model_size":self.getFileSize(os.path.join(self.model_dir,model_name+"-0000.params")),
                    "model_load_time_cpu":load_model_time
                }
            )
            mod,load_model_time = self.loadModel(self.model_dir, model_name),type="gpu")
            self.gpu_model_list.append(mod)
            self.gpu_model_time.append(
                {
                    "model_name":model_name,
                    "model_size":self.getFileSize(os.path.join(self.model_dir,model_name+"-0000.params")),
                    "model_load_time_gpu":load_model_time
                }
            )
        json.dump(self.cpu_model_time,open(os.path.join(self.result_dir,"gpu_model_time.json"),'w'),indent=2)
        json.dump(self.gpu_model_time,open(os.path.join(self.result_dir,"gpu_model_time.json"),'w'),indent=2)
    def testInference(self,times = 10000):
        inputQueue = Queue()
        outputQueue = Queue()
        processPool = []
        for i in range(gpu_num):
            processPool.append(Process(target=self.predictFromQueue,args=(inputQueue,outputQueue,i,)))
        for i in range(gpu_num):
            processPool[i].daemon = True
            processPool[i].start()

        predict_files = []
        for filename in os.listdir('../data'):
            if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
                predict_files.append(filename)
                img_data[filename] = get_image(filename,show=False)
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
