#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include  "cuda_kernel.h"

void cublasMatrix(const int64_t M,const int64_t K,const int64_t N,float *hostA, float *hostB, float *hostC)
{
    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cublasHandle_t handle; // cublas句柄
    cublasCreate(&handle); // 初始化句柄
    float alpha = 1.0;
    float beta = 0.0;
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
     cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, dA, K, dB, K, &beta, dC, M);
    // // cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //              N, M, K,
    //              &alpha,
    //              dB, CUDA_R_32F, N,
    //              dA, CUDA_R_32F, K,
    //              &beta,
    //              dC, CUDA_R_32F, N,
    //              CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

}
