# final_report from zfy
最终成绩performace.txt 1559.35GFlops,294.33x
## 优化过程
 1. 用cuda重写了winograd，分为filter变换GgGT，输入变换BTdB，和输出变换ATMT，其中U点乘V用到了cublas的sgemm函数。实现了260x的加速。

 2. 分析数据尺寸并结合L40的硬件规格，resize了blocksize和grid，取得了294x的加速

 3. 结合nsight-compute报告，发现cublassgemmbatched需要的寄存器太多，一个SM单元只能容纳三个block，只能取得33%occupation，尝试了减小batch的size，但是没有什么用，脑测了一下应该是矩阵的size才会影响寄存器的需要个数，但是为了能套到cublassgemmbatch里面，我在内存排布上面做了很多更改，如果要改矩阵size，要做出的牺牲有点大了，就没有进一步优化。nsight-sys的报告证明，优化kernel能取得的效果非常有限，amdahl定律最有用的一次。

 3. 结合nsight-sys报告，发现程序瓶颈在于IO操作，频繁的h2d和d2h数据移动拖累了整体程序，尤其是不必要的d2h写回操作，占据了cuda的72%运行时间。

 ![image1](/assets/nsight_sys.png "CUDA HW")

 为了进一步优化，本人尝试了四种办法。1.将每层的输出不写回host，保存在device端，读取下一层filter后进行计算，但是，阅读driver文件中val mode的代码，发现需要验证每层的输出，所以不得不将每层都写回到host。 2.使用pinned memory让io操作快一点，但是如果要使用pinned memory，就需要改动driver.cc,替换driver.cc的malloc函数，这是题目不允许的。3. 使用cuda内存池进行内存管理，我这样写了之后，程序加速比降为1065.43GFlops，201x。这种方式比手动cudamollac性能更差。4. 进行内存复用，将out保存在已经不用的input_image里面，但是这样子不能保证input的内存大小一定多于out，会出现segmentation fault。
## 测试说明
### 项目结构
* `driver.cc`是主程序入口，负责读取输入数据，调用 `winograd.cc` 中的函数进行计算，并输出结果。
* `Makefile` 是构建文件，用于编译整个项目。编译的优化选项用处也不大，留作纪念。
* `run.sh` 是用于提交 slurm 任务的脚本。`run_cuda.sh` 是用于提交nsight任务的脚本，不是运行程序的脚本。`val.sh` 是用于提交 slurm 正确性检查任务的脚本。
* `utils.cuh` 是cuda检查错误函数的声明。
* `utils.h` 是一些辅助函数的声明。
* `winograd.cu` 是我的主要优化代码
* `winograd.cc` 在原函数入口调用了声明为**extern C**的`winograd_cuda.cu`中的cuda函数
* `winograd.h` 是 `winograd.cc` 的头文件，`winograd_cuda.h` 是 `winograd_cuda.cu` 的头文件
* `env.sh` 是环境依赖的脚本，本优化实现需要依赖cuda@12.6.3
* `cuda_kernel.cu`和`cuda_kernel.h` 是最开始用cuda计算sgemm的优化，留作纪念
* `/cuda_test`是最开始测试环境用的代码，留作纪念
* `performance.txt`是在vgg16上所取得的性能最优结果，留作纪念
* `logs.md`是优化的日志，记录了一点思路和工作进度，留作纪念
### git tree
分支pure——cpu是在cpu上的优化代码，使用了OpenMP和SIMD进行优化，最终取得了不到40x，本人对cpu优化实在是毫无经验。之前有一点点写过cuda代码的经验，就转投cuda了，最后得到的效果优于我自己的cpu优化。不过看到绝密资料中cpu优化达到了2000x，实在是叹为观止。
### how to build and run
    ```bash
    cd /recruitment-2025-spring-zfy
    source env.sh
    make
    sbatch run.sh
    ```

因为存在warm-up的问题，输出1550GFlops左右都有可能，多次提交后的输出会更高。
## 参考和致谢
### reference

* 代码思路，内存排布参考了 https://zhuanlan.zhihu.com/p/260109670

* 小部分代码和优化思路来源于 https://github.com/xuqiantong/CUDA-Winograd/blob/master/Kernel256_winograd.cu， https://github.com/UDC-GAC/openCNN， https://github.com/Sha-x2-nk/WinogradConvolution-CUDA， https://github.com/md2z34/winograd_cuda/tree/master/cpu。

* 在sgemm部分使用了 cuBLAS cublasSgemmStridedBatched函数。

* 基本没有用到ai生成的代码，换了deepseek都写不明白复杂线程网络的cuda和cublas的参数，最后还是自己画了一下午矩阵到底是什么形状，内存中是什么样子，然后调出来的。用到了ai指导如何使用vtune,nsight和makefile之类的工具，感觉不是很关键，暂且不表。
### acknowledge
这是第一次真正意义上接触比较真实的HPC，虽然比较toy，但是和我想象的差不多，很开心，没有白学。感谢七边形超算队，本人接触HPC的契机。