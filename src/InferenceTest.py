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
batchsize = 1
Batch = namedtuple('Batch', ['data'])
model_names = ["caffenet","Inception-BN","nin","resnet-152","RN101-5k500","squeezenet_v1.1","vgg16","vgg19"]

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
def load_model(name,type="cpu"):
    begin = time.time()
    sym, arg_params, aux_params = mx.model.load_checkpoint(name, 0)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None) if type=="cpu" else mx.mod.Module(symbol=sym, context=mx.gpu(1), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (batchsize, 3, 224, 224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    end = time.time()
    return mod,end-begin
def predict(names,mod):
    imgs = []
    for name in names:
        img = get_image(name, show=False)
        imgs.append(img)
    begin = time.time()
    mod.forward(Batch([mx.nd.array(imgs)]))
    prob = mod.get_outputs()[0].asnumpy()
    end = time.time()
    return end-begin
def get_FileSize(filePath):
    filePath = unicode(filePath,'utf8')
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024*1024)
    return round(fsize,2)
if __name__ == "__main__":
    path_dir = "../model/"
    cpu_model_list = []
    gpu_model_list = []
    cpu_model_time = []
    gpu_model_time = []
    for model_name in model_names:
        mod,load_model_time = load_model(path_dir + model_name,type="cpu")
        cpu_model_list.append(mod)
        cpu_model_time.append(
            {
                "model_name":model_name,
                "model_size":get_FileSize("../model/"+model_name+"-0000.params"),
                "model_load_time_cpu":load_model_time
            }
        )
        mod,load_model_time = load_model(path_dir + model_name,type="gpu")
        gpu_model_list.append(mod)
        gpu_model_time.append(
            {
                "model_name":model_name,
                "model_size":get_FileSize("../model/"+model_name+"-0000.params"),
                "model_load_time_gpu":load_model_time
            }
        )
    json.dump(cpu_model_time,open("../cpu_model_time.json",'w'),indent=2)
    json.dump(gpu_model_time,open("../gpu_model_time.json",'w'),indent=2)
    predict_files = []
    for filename in os.listdir('../data'):
        if filename.endswith("jpg") or filename.endswith("jpeg") or filename.endswith(".png") or filename.endswith(".gif"):
            predict_files.append(filename)
    # for name in predict_files:
    #     try:
    #         img = get_image(name)
    #         print img.shape
    #     except Exception,ex:
    #         os.remove("../data/"+name)
    #         print name
    # exit()
    times = 10000
    cpu_infer_time = []
    gpu_infer_time = []
    for i in range(times):
        sampled = random.sample(predict_files,batchsize)
        for j in range(len(cpu_model_time)):
            t = predict(sampled,cpu_model_list[j])
            cpu_infer_time.append(
                {
                    "model_name":cpu_model_time[j]["model_name"],
                    "data_set":i,
                    "batchsize":batchsize,
                    "cpu_time":t
                }
            )
            print i,cpu_model_time[j]["model_name"],t
        for j in range(len(gpu_model_time)):
            t = predict(sampled,gpu_model_list[j])
            gpu_infer_time.append(
                {
                    "model_name":gpu_model_time[j]["model_name"],
                    "data_set":i,
                    "batchsize":batchsize,
                    "gpu_time":t
                }
            )
            print i,gpu_model_time[j]["model_name"],t
    json.dump(cpu_infer_time,open("../cpu_infer_time_%d.json"%batchsize,'w'),indent=2)
    json.dump(gpu_infer_time,open("../gpu_infer_time_%d.json"%batchsize,'w'),indent=2)
                    # os.remove('./data/'+filename)
