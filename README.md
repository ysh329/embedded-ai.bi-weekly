# Awesome-Emebedded-AI
<p align="center">
  <a href="https://github.com/ysh329/Embedded-AI-awesome">
    <img alt="Embedded-AI-awesome" src="logo.jpg" width="300">
  </a>
</p>

<div align="center">

<p align="center">
  A curated list of awesome <a href="#">A.I.</a> & <a href="#">Embedded/Mobile-devices</a> resources, tools and more.
</p>

<p align="center">
  <a href="https://github.com/ysh329/Embedded-AI-awesome"><img alt="Awesome Badge" src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg"></a>
  <a href="https://github.com/ysh329/Embedded-AI-awesome/pulls"><img alt="Pull Requests Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square"></a>
  <a href="https://gitter.im/embedded_ai"><img alt="Chat on Gitter" src="https://badges.gitter.im/tobiasbueschel/awesome-pokemon.svg"></a>
</p>

<p>
<i>Looking for contributors. Submit a pull request if you have something to add :) </i><br>  
Please check the <a href="https://github.com/ysh329/Embedded-AI-awesome/blob/master/contributing.md">contribution guidelines</a> for info on formatting and writing pull requests.
</p>

</div>

## Contents

- [Embedded-AI-paper-list](#Embedded\-AI\-paper\-list)
- [Embedded-AI-App-experience]  
- [Embedded-AI-demos](#Embedded-AI-demos)
  - [computer-vision-demos](#computer-vision-demos)
  - [speech-demos](#speech-demos)
  - [natural-language-process-demos](#nlp-demos)
- [Embedded-AI-frameworks](#Embedded-AI-frameworks)
  - [inference-frameworks](#inference-frameworks)
  - [benchmark-frameworks](#benchmark-frameworks)
- [Embedded-benchmark]
- [News]

## Embedded-AI-paper-list
Embedded-AI-paper-list

### Classic

* [1512.03385] Deep Residual Learning for Image Recognition  
https://arxiv.org/abs/1512.03385

* [1610.02357] Xception: Deep Learning with Depthwise Separable Convolutions  
https://arxiv.org/abs/1610.02357

* [1611.05431] **ResneXt**: Aggregated Residual Transformations for Deep Neural Networks  
https://arxiv.org/abs/1611.05431

### Overview

* [1707.01209] Model compression as constrained optimization, with application to neural nets. Part I: general framework  
https://arxiv.org/abs/1707.01209

* [1707.04319] Model compression as constrained optimization, with application to neural nets. Part II: quantization  
https://arxiv.org/abs/1707.04319

### Representation

* [SenSys ’16]Sparsification and Separation of Deep Learning Layers for Constrained Resource Inference on Wearables  
http://niclane.org/pubs/sparsesep_sensys.pdf

* [IoT-App ’15]An Early Resource Characterization of Deep Learning on Wearables, Smartphones and Internet-of-Things Devices  
http://niclane.org/pubs/iotapp15_early.pdf

### Structure/Pattern

* [1707.06342] ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression  
https://arxiv.org/abs/1707.06342

* [1707.01083] ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
https://arxiv.org/abs/1707.01083

* [1704.04861] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/abs/1704.04861

* [1706.03912] SEP-Nets: Small and Effective Pattern Networks  
https://arxiv.org/abs/1706.03912

### Binarization

* [1707.04693] Binarized Convolutional Neural Networks with Separable Filters for Efficient Hardware Acceleration  
https://arxiv.org/abs/1707.04693

* [1602.02830] Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1  
https://arxiv.org/abs/1602.02830

* [1603.05279] XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks
https://arxiv.org/abs/1603.05279

  * [1705.09864] BMXNet: An Open-Source Binary Neural Network Implementation Based on MXNet
https://arxiv.org/abs/1705.09864

* [1606.06160] DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
https://arxiv.org/abs/1606.06160

### Distillation

* [1503.02531] Distilling the Knowledge in a Neural Network  
https://arxiv.org/abs/1503.02531

* Face Model Compression by Distilling Knowledge from Neurons  
http://www.ee.cuhk.edu.hk/~xgwang/papers/luoZLWXaaai16.pdf

### Pruning

* [NIPS'15] Learning both Weights and Connections for Efficient Neural Networks  
https://arxiv.org/abs/1506.02626 

* [ICLR'17] Pruning Filters for Efficient ConvNets  
https://arxiv.org/abs/1608.08710

* [ICLR'17] Pruning Convolutional Neural Networks for Resource Efficient Inference  
https://arxiv.org/abs/1611.06440

* [ICLR'17] Soft Weight-Sharing for Neural Network Compression  
https://arxiv.org/abs/1702.04008

* [ICLR'16] Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding  
https://arxiv.org/abs/1510.00149

* [NIPS'16] Dynamic Network Surgery for Efficient DNNs  
https://arxiv.org/abs/1608.04493

* [CVPR'17] Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning  
https://arxiv.org/abs/1611.05128

### Quantization

* [ICML'17] The ZipML Framework for Training Models with End-to-End Low Precision: The Cans, the Cannots, and a Little Bit of Deep Learning  
https://arxiv.org/abs/1611.05402

* [1412.6115] Compressing Deep Convolutional Networks using Vector Quantization  
https://arxiv.org/abs/1412.6115

* [CVPR '16] Quantized Convolutional Neural Networks for Mobile Devices  
https://arxiv.org/abs/1512.06473

* [ICASSP'16] Fixed-Point Performance Analysis of Recurrent Neural Networks  
https://arxiv.org/abs/1512.01322

* [arXiv'16] Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations  
https://arxiv.org/abs/1609.07061

* [ICLR'17] Loss-aware Binarization of Deep Networks  
https://arxiv.org/abs/1611.01600

* [ICLR'17] Towards the Limit of Network Quantization  
https://arxiv.org/abs/1612.01543

* [CVPR'17] Deep Learning with Low Precision by Half-wave Gaussian Quantization  
https://arxiv.org/abs/1702.00953

* [1706.02393] ShiftCNN: Generalized Low-Precision Architecture for Inference of Convolutional Neural Networks  
https://arxiv.org/abs/1706.02393) 

#### Low Rank Approximation

* [CVPR'15] Efficient and Accurate Approximations of Nonlinear Convolutional Networks  
https://arxiv.org/abs/1411.4229

  * Accelerating Very Deep Convolutional Networks for Classification and Detection  
  https://arxiv.org/abs/1505.06798
  
* [1511.06067] Convolutional neural networks with low-rank regularization  
https://arxiv.org/abs/1511.06067

* [NIPS'14] Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation  
https://arxiv.org/abs/1404.0736

* [ICLR'16] Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications  
https://arxiv.org/abs/1511.06530

### Execution/Frameworks

* [1605.04614] DeepLearningKit - an GPU Optimized Deep Learning Framework for Apple's iOS, OS X and tvOS developed in Metal and Swift  
https://arxiv.org/abs/1605.04614

* [MobiSys '17] DeepMon: Mobile GPU-based Deep Learning Framework for Continuous Vision Applications]  
https://www.sigmobile.org/mobisys/2017/accepted.php 

* [MobiSys '17] DeepEye: Resource Efficient Local Execution of Multiple Deep Vision Models using Wearable Commodity Hardware  
http://fahim-kawsar.net/papers/Mathur.MobiSys2017-Camera.pdf

* [EMDL '17] MobiRNN: Efficient Recurrent Neural Network Execution on Mobile GPU  
https://arxiv.org/abs/1706.00878) 

* [WearSys '16] DeepSense: A GPU-based deep convolutional neural network framework on commodity mobile devices  
http://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=4278&context=sis_research

* [IPSN '16] DeepX: A Software Accelerator for Low-Power Deep Learning Inference on Mobile Devices  
http://niclane.org/pubs/deepx_ipsn.pdf

* [ISCA '16] EIE: Efficient Inference Engine on Compressed Deep Neural Network  
https://arxiv.org/abs/1602.01528

* [MobiSys '16] MCDNN: An Approximation-Based Execution Framework for Deep Stream Processing Under Resource Constraints  
http://haneul.github.io/papers/mcdnn.pdf

* [MobiCASE '16] DXTK: Enabling Resource-efficient Deep Learning on Mobile and Embedded Devices with the DeepX Toolkit  
http://niclane.org/pubs/dxtk_mobicase.pdf

* [MM '16] CNNdroid: GPU-Accelerated Execution of Trained Deep Convolutional Neural Networks on Android  
https://arxiv.org/abs/1511.07376



## Embedded-AI-App-experience

* 【技术共享】怎么把人脸检测的速度做到极致  
https://mp.weixin.qq.com/s?__biz=MzA3NDU3MTc1Ng==&mid=2651165778&idx=1&sn=2f2d8f6b7a11d381a4290a20817b46a2

* 基于OpenGL ES 的深度学习框架编写 - jxt1234and2010的专栏 - CSDN博客  
http://blog.csdn.net/jxt1234and2010/article/details/71056736

# Codes

## Mobile App Examples

### Android
 
* KleinYuan/Caffe2-iOS: Caffe2 on iOS Real-time Demo. Test with Your Own Model and Photos.  
https://github.com/KleinYuan/Caffe2-iOS

* MXNet Android Classification App - Image classification on Android with MXNet.  
https://github.com/Leliana/WhatsThis

### iOS

* MXNet iOS Classification App - Image classification on iOS with MXNet.  
https://github.com/pppoe/WhatsThis-iOS

* Compile MXnet on Xcode (in Chinese) - a step-by-step tutorial of compiling MXnet on Xcode for iOS app  
http://www.liuxiao.org/2015/12/ios-mxnet-%E7%9A%84-ios-%E7%89%88%E6%9C%AC%E7%BC%96%E8%AF%91/

* KimDarren/FaceCropper: Crop faces, inside of your image, with iOS 11 Vision api.
https://github.com/KimDarren/FaceCropper

### FPGA

## OpenCL/OpenGL/Vulkan/RenderScript/ARMComputeLibrary

## Embedded-AI-frameworks

* jiaxiang-wu/quantized-cnn: An efficient framework for convolutional neural networks  
https://github.com/jiaxiang-wu/quantized-cnn

## Course

* [Deep learning **systems**](http://dlsys.cs.washington.edu/schedule), UW course schedule(focused on systems design, not learning)

## Guides/Tutorials

* [Squeezing Deep Learning Into Mobile Phones](https://www.slideshare.net/anirudhkoul/squeezing-deep-learning-into-mobile-phones)

* [Deep Learning – Tutorial and Recent Trends](https://www.dropbox.com/s/p7lvelt0aihrwtl/FPGA%2717%20tutorial%20Song%20Han.pdf?dl=0)

* [Efficient Convolutional Neural Network Inference on Mobile GPUs](https://www.slideshare.net/embeddedvision/efficient-convolutional-neural-network-inference-on-mobile-gpus-a-presentation-from-imagination-technologies)

## News

* We ported CAFFE to HIP - and here’s what happened… - GPUOpen  
http://gpuopen.com/ported-caffe-hip-heres-happened/

* Clarifai launches SDK for training AI on your iPhone | VentureBeat | AI | by Khari Johnson
https://venturebeat.com/2017/07/12/clarifai-launches-sdk-for-running-ai-on-your-iphone/
