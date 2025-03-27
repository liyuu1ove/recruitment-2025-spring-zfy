#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>

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

void sgemm_cublas(const int64_t M, const int64_t N, const int64_t K, float *A, float *B, float *C) {
    float alpha = 1.0f, beta = 0.0f;

    typedef float(*A_tensor_t)[K];
    typedef float(*B_tensor_t)[K];
    typedef float(*C_tensor_t)[M];
    A_tensor_t A_tensor = (A_tensor_t)A;
    B_tensor_t B_tensor = (B_tensor_t)B;
    C_tensor_t C_tensor = (C_tensor_t)C;

    // 初始化cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // 将数据从主机复制到设备
    cublasSetMatrix(M, K, sizeof(float), A_tensor, K, d_A, K);
    cublasSetMatrix(N, K, sizeof(float), B_tensor, K, d_B, K);
    cublasSetMatrix(N, M, sizeof(float), C_tensor, M, d_C, M);

    // 执行SGEMM，注意B和C已经被转置
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha, d_A, M, d_B, N, &beta, d_C, M);

    // 将结果从设备复制回主机
    //cublasGetMatrix(M, N, sizeof(float), d_C, M, C_tensor, M);
    cudaMemcpy(C_tensor,d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 销毁cuBLAS句柄
    cublasDestroy(handle);
}
