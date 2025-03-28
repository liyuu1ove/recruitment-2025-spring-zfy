### 2025年3月24日16点11分
    折腾了一下cpu优化，使用OpenMP和avx512达到了35x，看vtune里面提示sgemm占用50%多，点开之后向量相乘的汇编占了50%多，感觉优化空间不是很大。
### 2025年3月25日10点24分
    问ai如何使用OpenMP进一步优化，有的选项连编译都不能过。看vtune里面提示pipeline slot利用率太低了，也不知道怎么优化，感觉可以转投cuda了。
### 2025年3月26日22点46分
    折腾了一天nvcc和spack
### 2025年3月27日20点21分
    上午解决了编译的依赖问题，结果一跑代码过不了val，哈哈。
    用cublas优化了一下矩阵乘法，cublassgemm的参数真难理解啊，在前面又做了packing，那几个参数再也搞不明白是哪个了。不用cublasgetmatrix之后，终于想通了，转置之后的矩阵主维度应该改一下的，下午才过了val，结果一跑vgg16才3.5x。脑测应该是频繁加载数据的问题，在那个循环里面也没办法并行，用这个框架感觉很难实现局部的cuda加速，准备用cuda重写整个代码，就可以换batch版的cublassgemm了。晚上写了filter_transform和image_transform，实现起来没什么难度，参考了github https://github.com/xuqiantong/CUDA-Winograd/blob/master/Kernel256_winograd.cu 里面的写法，不过我直接把filter的形状写死了，并做了任意输入的适配，不过没有做不能完美分块的适配，edgecase直接OpenCNN https://github.com/UDC-GAC/openCNN 和有一篇论文里面提到F（6*6，3*3）是最高效的分块策略，就这样写了
### 2025年3月28日19点22分
    上午看到知乎上详解winograd里面提到了cuDNN的lib，本来以为能参考一下，结果cuDNN是header only的lib，遂放弃了https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html#cudnnconvolutionforward
    下午写完了，编译的时候没什么报错，但是始终过不了val，最后发现是写死的BT矩阵没有做转置，参数错了。没有做很多的适配，只是用cuda重写了一下，参考了cuda编程指南的线程分配策略，随便试了几个参数，达到了260x的加速比，上午看到集群中是两个L40，明天看看能不能针对这个架构做优化什么的吧。
    