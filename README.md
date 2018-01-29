# CortexInferencePerformanceTest
Test described in whitepaper
## Environment

### Hardware
* CPU: E5-2683 v3
* GPU: 8x1080Ti
* RAM: 64 GB
* Disk: SSD 60 EVO 250 GB

### Software
* MXNet
* OpenCV 3.1

### Others
* Dataset: ImageNet Dataset (winter 2011 release) [http://image-net.org/imagenet_data/urls/imagenet_winter11_urls.tgz]
* Models [http://mxnet.incubator.apache.org/model_zoo/index.html]:  
  * CaffeNet
  * Network in Network
  * SqueezeNet
  * VGG16
  * VGG19
  * Inception v3 / BatchNorm
  * ResNet-152
  * ResNet101-64x4d

## Usage
* src/InferenceTest.py is single GPU test for inference.
* src/InferenceStreamTest.py is multi-GPU test for image stream data inference.
* src/config.json is parameters.

### config.json
* models     : model names
* img_dir    : images directory (put images into this directory)
* result_dir : results directory
* model_dir  : models directory (put models into this directory)
* gpu_num    : gpu # for inference
* batchsize  : batchsize for inference
* times      : number of times for inference test and number of images for inference stream test
