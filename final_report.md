
### 项目结构

* `driver.cc`是主程序入口，负责读取输入数据，调用 `winograd.cc` 中的函数进行计算，并输出结果。
* `Makefile` 是构建文件，用于编译整个项目。编译的优化选项用处也不大，留作纪念。
* `run.sh` 是用于提交 slurm 任务的脚本。`run_cuda.sh` 是用于提交nsight任务的脚本，不是运行程序的脚本。`val.sh` 是用于提交 slurm 正确性检查任务的脚本。
* `utils.cuh` 是cuda检查错误函数的声明。
* `utils.h` 是一些辅助函数的声明。
* `winograd.cu` 是我的主要优化代码
* `winograd.cc` 在原函数入口调用了声明为C接口的`winograd_cuda.cu`中的cuda函数
* `winograd.h` 是 `winograd.cc` 的头文件，`winograd_cuda.h` 是 `winograd_cuda.cu` 的头文件
* `env.sh` 是环境依赖的脚本，本优化实现需要依赖cuda@12.6.3
* `cuda_kernel.cu`和`cuda_kernel.h` 是最开始用cuda计算sgemm的优化，留作纪念
* `/cuda_test`是最开始测试环境用的代码，留作纪念
* `performance.txt`是在vgg16上所取得的性能最优结果，留作纪念
* `logs.md`是优化的日志，记录了一点思路和工作进度，留作纪念

### how to build and run
    ```bash
    cd /recruitment-2025-spring-zfy
    source env.sh
    make
    sbatch run.sh
    ```

因为存在warm-up的问题，输出1550GFlops左右都有可能，多次提交后的输出会更高。
### reference

* 代码思路，内存排布参考了 https://zhuanlan.zhihu.com/p/260109670

* 部分代码和优化思路来源于 https://github.com/xuqiantong/CUDA-Winograd/blob/master/Kernel256_winograd.cu， https://github.com/UDC-GAC/openCNN， https://github.com/Sha-x2-nk/WinogradConvolution-CUDA， https://github.com/md2z34/winograd_cuda/tree/master/cpu。

* 在sgemm部分使用了 cuBLAS cublasSgemmStridedBatched函数。

* 基本没有用到ai生成的代码，换了deepseek都写不明白复杂线程网络的cuda和cublas的参数，最后还是自己画了一下午图调出来的。用到了ai指导如何使用nsight和makefile，感觉不是很关键，暂且不表。
### acknowledge

