#ifndef ERROR_CHECK_UTILS_CUH
#define ERRORCHECKUTILS_CUH

#include <cuda_runtime.h>

#include <stdio.h>

// cuda error checking macro
#define CUDA_CALL(x)                                             \
    do                                                           \
    {                                                            \
        cudaError_t err = (x);                                   \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("CUDA error at %s:%d - %s\n",                 \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            printf("Failed CUDA API call: %s\n", #x);            \
        }                                                        \
    } while (0)

#define CUBLAS_CALL(call)                                   \
    {                                                       \
        cublasStatus_t cublasError = call;                  \
        if (cublasError != CUBLAS_STATUS_SUCCESS)           \
        {                                                   \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", \
                    __FILE__, __LINE__, cublasError);       \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    }

#endif
//检查错误宏，没什么性能太大影响
