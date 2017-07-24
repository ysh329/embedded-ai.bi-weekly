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

- [Papers](#papers)  
  - [Classic](#classic)
  - [Overview](#overview)
  - [Representation](#representation)
  - [Structure/Pattern](#structure)
  - [Binarization](#binarization)
  - [Pruning](#pruning)
  - [Quantization](#quantization)
  - [Low Rank Approximation](#lowrankapproximation)
  - [Distillation](#distillation)
  - [Execution/Frameworks](#frameworkpaper)
- [App-Experience](#experience)  
- [Demo-Codes](#codes)
  - [Android](#android)
  - [iOS](#ios)
  - [Vulkan](#vulkan)
- [Frameworks](#frameworks)
  - [general](#general)
  - [inference](#inference)
  - [benchmark](#benchmark)
- [Course/Guide/Tutorial](#course)
- [News](#news)

# Papers

## Classic

* [1512.03385] Deep Residual Learning for Image Recognition  
https://arxiv.org/abs/1512.03385

* [1610.02357] Xception: Deep Learning with Depthwise Separable Convolutions  
https://arxiv.org/abs/1610.02357

* [1611.05431] **ResneXt**: Aggregated Residual Transformations for Deep Neural Networks  
https://arxiv.org/abs/1611.05431

## Overview

* [1707.01209] Model compression as constrained optimization, with application to neural nets. Part I: general framework  
https://arxiv.org/abs/1707.01209

* [1707.04319] Model compression as constrained optimization, with application to neural nets. Part II: quantization  
https://arxiv.org/abs/1707.04319

## Representation

* [SenSys ’16] Sparsification and Separation of Deep Learning Layers for Constrained Resource Inference on Wearables  
http://niclane.org/pubs/sparsesep_sensys.pdf

* [IoT-App ’15] An Early Resource Characterization of Deep Learning on Wearables, Smartphones and Internet-of-Things Devices  
http://niclane.org/pubs/iotapp15_early.pdf

## Structure

* [1707.06342] ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression  
https://arxiv.org/abs/1707.06342

* [1707.01083] ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
https://arxiv.org/abs/1707.01083

* [1704.04861] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
https://arxiv.org/abs/1704.04861

* [1706.03912] SEP-Nets: Small and Effective Pattern Networks  
https://arxiv.org/abs/1706.03912

## Binarization

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

## Pruning

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

## Quantization

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

## LowRankApproximation

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

## Distillation

* [1503.02531] Distilling the Knowledge in a Neural Network  
https://arxiv.org/abs/1503.02531

* Face Model Compression by Distilling Knowledge from Neurons  
http://www.ee.cuhk.edu.hk/~xgwang/papers/luoZLWXaaai16.pdf

## FrameworkPaper

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

# Experience

* 【技术共享】怎么把人脸检测的速度做到极致  
https://mp.weixin.qq.com/s?__biz=MzA3NDU3MTc1Ng==&mid=2651165778&idx=1&sn=2f2d8f6b7a11d381a4290a20817b46a2

* 基于OpenGL ES 的深度学习框架编写 - jxt1234and2010的专栏 - CSDN博客  
http://blog.csdn.net/jxt1234and2010/article/details/71056736

# Codes

## Android

* harvardnlp/nmt-android: Neural Machine Translation on Android](https://github.com/harvardnlp/nmt-android

* TensorFlow Android Camera Demo  
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android

* KleinYuan/Caffe2-iOS: Caffe2 on iOS Real-time Demo. Test with Your Own Model and Photos.  
https://github.com/KleinYuan/Caffe2-iOS

* MXNet Android Classification App - Image classification on Android with MXNet.  
https://github.com/Leliana/WhatsThis

* bwasti/AICamera: Demonstration of using Caffe2 inside an Android application.  
https://github.com/bwasti/AICamera

* mtmd/Mobile_ConvNet: RenderScript based implementation of Convolutional Neural Networks for Android phones  
https://github.com/mtmd/Mobile_ConvNet

## iOS

* MXNet iOS Classification App - Image classification on iOS with MXNet.  
https://github.com/pppoe/WhatsThis-iOS

* Compile MXnet on Xcode (in Chinese) - a step-by-step tutorial of compiling MXnet on Xcode for iOS app  
http://www.liuxiao.org/2015/12/ios-mxnet-%E7%9A%84-ios-%E7%89%88%E6%9C%AC%E7%BC%96%E8%AF%91/

* KleinYuan/Caffe2-iOS: Caffe2 on iOS Real-time Demo. Test with Your Own Model and Photos.
https://github.com/KleinYuan/Caffe2-iOS

* KimDarren/FaceCropper: Crop faces, inside of your image, with iOS 11 Vision api.
https://github.com/KimDarren/FaceCropper

* hollance/TensorFlow-iOS-Example: Source code for my blog post "Getting started with TensorFlow on iOS"  
https://github.com/hollance/TensorFlow-iOS-Example

## Vulkan

* SaschaWillems/Vulkan: Examples and demos for the new Vulkan API  
https://github.com/SaschaWillems/Vulkan

* ARM-software/vulkan-sdk: ARM Vulkan SDK  
https://github.com/ARM-software/vulkan-sdk

* alexhultman/libvc: Vulkan Compute for C++ (experimentation project)  
https://github.com/alexhultman/libvc

# Frameworks

## General  
General frameworks contain inference and backprop stages.

## Inference  
Inference frameworks contains inference stage only.

* Deep Learning in a Single File for Smart Devices — mxnet  
https://github.com/dmlc/mxnet/tree/master/amalgamation

* ARM-software/ComputeLibrary: The ARM Computer Vision and Machine Learning library is a set of functions optimised for both ARM CPUs and GPUs using SIMD technologies  
https://github.com/ARM-software/ComputeLibrary  
[Intro](https://developer.arm.com/technologies/compute-library)

* Apple CoreML  
https://developer.apple.com/documentation/coreml

* Microsoft Embedded Learning Library  
https://github.com/Microsoft/ELL

* mil-tokyo/webdnn: Fastest DNN Execution Framework on Web Browser  
https://github.com/mil-tokyo/webdnn

* jiaxiang-wu/quantized-cnn: An efficient framework for convolutional neural networks  
https://github.com/jiaxiang-wu/quantized-cnn

* Tencent/ncnn: ncnn is a high-performance neural network inference framework optimized for the mobile platform  
https://github.com/Tencent/ncnn

## Benchmark

* baidu-research/DeepBench: Benchmarking Deep Learning operations on different hardware  
https://github.com/baidu-research/DeepBench

* facebook

## Convertor

Model convertor. more convertos please refer [deep-learning-model-convertor  
https://github.com/ysh329/deep-learning-model-convertor)

* NervanaSystems/caffe2neon: Tools to convert Caffe models to neon's serialization format  
https://github.com/NervanaSystems/caffe2neon

# Course

This part contains related course, guides and tutorials.

* Deep learning **systems**: UW course schedule(focused on systems design, not learning)  
http://dlsys.cs.washington.edu/schedule

* Squeezing Deep Learning Into Mobile Phones  
https://www.slideshare.net/anirudhkoul/squeezing-deep-learning-into-mobile-phones

* Deep Learning – Tutorial and Recent Trends  
https://www.dropbox.com/s/p7lvelt0aihrwtl/FPGA%2717%20tutorial%20Song%20Han.pdf?dl=0

* Efficient Convolutional Neural Network Inference on Mobile GPUs  
https://www.slideshare.net/embeddedvision/efficient-convolutional-neural-network-inference-on-mobile-gpus-a-presentation-from-imagination-technologies

* ARM® Mali™ GPU OpenCL Developer Guide  
html: http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.100614_0303_00_en/ada1432742770595.html  
pdf:  http://infocenter.arm.com/help/topic/com.arm.doc.100614_0303_00_en/arm_mali_gpu_opencl_developer_guide_100614_0303_00_en.pdf

* Optimal Compute on ARM MaliTM GPUs  
http://www.cs.bris.ac.uk/home/simonm/montblanc/OpenCL_on_Mali.pdf

* GPU Compute for Mobile Devices  
http://www.iwocl.org/wp-content/uploads/iwocl-2014-workshop-Tim-Hartley.pdf

* Compute for Mobile Devices Performance focused  
http://kesen.realtimerendering.com/Compute_for_Mobile_Devices5.pdf

* Hands On OpenCL  
https://handsonopencl.github.io/

* Adreno OpenCL Programming Guide  
https://developer.qualcomm.com/download/adrenosdk/adreno-opencl-programming-guide.pdf

* Better OpenCL Performance on Qualcomm Adreno GPU  
https://developer.qualcomm.com/blog/better-opencl-performance-qualcomm-adreno-gpu-memory-optimization

## News

* We ported CAFFE to HIP - and here’s what happened… - GPUOpen  
http://gpuopen.com/ported-caffe-hip-heres-happened/

* Clarifai launches SDK for training AI on your iPhone | VentureBeat | AI | by Khari Johnson
https://venturebeat.com/2017/07/12/clarifai-launches-sdk-for-running-ai-on-your-iphone/
