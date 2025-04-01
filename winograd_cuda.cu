#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <omp.h>

#include "utils.cuh"
#include "winograd_cuda.h"



__device__ __forceinline__ void multiply_G(const float in0,const float in1,const float in2,
    float& out0,float& out1,float& out2,float& out3,float& out4,float& out5)
{
    // G = [
    //      1/4.0,      0,       0,
    //     -1/6.0,  -1/6.0, -1/6.0,
    //     -1/6.0,   1/6.0, -1/6.0,
    //     1/24.0,  1/12.0,  1/6.0,
    //     1/24.0, -1/12.0,  1/6.0,
    //          0,       0,      1
    // ]

    float temp[4]={(-in0 - in2)/6.0f,in0 / 24.0f,in1 / 12.0f,in2 / 6.0f};

    out0 = in0 / 4.0f;
    out1 = temp[0] - 2*temp[2];
    out2 = temp[0] + 2*temp[2];
    out3 = temp[1] + temp[2] + temp[3];
    out4 = temp[1] - temp[2] + temp[3];
    out5 = in2;
}

template <int NUM_KERNELS_PER_BLOCK, int BLOCK_SIZE>
__global__ void winograd_GgGT(
    const float *g, const int K, const int C,
    float *filter_transform)
{

    const int NUM_KERNELS = K * C;
    const int kc_idx_start = blockIdx.x * NUM_KERNELS_PER_BLOCK;

    __shared__ float shared_6x6[NUM_KERNELS_PER_BLOCK][6][6];

    for (int i = threadIdx.x; i < NUM_KERNELS_PER_BLOCK * 3 * 3; i += BLOCK_SIZE)//read from g
    {
        const int col_idx = i % 3;
        const int row_idx = (i / 3) % 3;
        const int kc_idx = (i / 9);

        const int global_kc = kc_idx + kc_idx_start;
        if (global_kc < NUM_KERNELS)
            shared_6x6[kc_idx][row_idx][col_idx] = g[global_kc * 9 + row_idx * 3 + col_idx];
    }
    __syncthreads();

    // Gg
    __shared__ float shared_6x3[NUM_KERNELS_PER_BLOCK][6][3];

    for (int i = threadIdx.x; i < NUM_KERNELS_PER_BLOCK * 3; i += BLOCK_SIZE)
    {
        //col-wise g
        const int col_idx = i % 3;
        const int kc_idx = i / 3;

        if (kc_idx_start + kc_idx < NUM_KERNELS)
        {
            multiply_G(shared_6x6[kc_idx][0][col_idx],shared_6x6[kc_idx][1][col_idx],shared_6x6[kc_idx][2][col_idx],
                    shared_6x3[kc_idx][0][col_idx],shared_6x3[kc_idx][1][col_idx],shared_6x3[kc_idx][2][col_idx],shared_6x3[kc_idx][3][col_idx],shared_6x3[kc_idx][4][col_idx],shared_6x3[kc_idx][5][col_idx]);
        }
    }
    __syncthreads();

    // GgGT
    for (int i = threadIdx.x; i < NUM_KERNELS_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        // row-wise Gg
        const int row_idx = i % 6;
        const int kc_idx = i / 6;

        if (kc_idx + kc_idx_start < NUM_KERNELS)
        {
            multiply_G(shared_6x3[kc_idx][row_idx][0],shared_6x3[kc_idx][row_idx][1],shared_6x3[kc_idx][row_idx][2],
                 shared_6x6[kc_idx][row_idx][0], shared_6x6[kc_idx][row_idx][1], shared_6x6[kc_idx][row_idx][2], shared_6x6[kc_idx][row_idx][3], shared_6x6[kc_idx][row_idx][4], shared_6x6[kc_idx][row_idx][5]);
        }
    }
    __syncthreads();

    // loading back to GMem
    for (int i = threadIdx.x; i < 6 * 6 * NUM_KERNELS_PER_BLOCK; i += BLOCK_SIZE)
    {
        const int kc_idx = i % NUM_KERNELS_PER_BLOCK;
        const int offset = i / NUM_KERNELS_PER_BLOCK;

        const int col_idx = offset % 6;
        const int row_idx = offset / 6;

        const int global_kc = kc_idx_start + kc_idx;
        if (global_kc < NUM_KERNELS)
            filter_transform[offset * NUM_KERNELS + global_kc] = shared_6x6[kc_idx][row_idx][col_idx];
    }
}

__device__ __forceinline__ void multiply_BT(const float in0,const float in1,const float in2,const float in3,const float in4,const float in5,
    float& out0,float& out1,float& out2,float& out3,float& out4,float& out5)
{
    /*
        BT = [
                4,  0, -5,  0, 1, 0,
                0, -4, -4,  1, 1, 0,
                0,  4, -4, -1, 1, 0,
                0, -2, -1,  2, 1, 0,
                0,  2, -1, -2, 1, 0,
                0,  4,  0, -5, 0, 1
             ]
    */
    float temp[4]={in3 - 4 * in1,in4 - 4 * in2,2 * (in1 - in3),in4 - in2};

    out0 = 4 * in0 - 5 * in2 + in4;
    out1 = temp[0] + temp[1];
    out2 = temp[1] - temp[0];
    out3 = temp[3] - temp[2];
    out4 = temp[2] + temp[3];
    out5 = 4 * in1 - 5 * in3 + in5;
}

template <int TILES_H_PER_BLOCK, int TILES_W_PER_BLOCK, int BLOCK_SIZE>
__global__ void winograd_BTdB(
    const float *d, const int N, const int C, const int H, const int W,
    float *image_transform, const int TILES_H, const int TILES_W)//image_transform
{

    const int tile_col_idx_start = blockIdx.x * TILES_W_PER_BLOCK;
    const int tile_row_idx_start = blockIdx.y * TILES_H_PER_BLOCK;
    const int nc = blockIdx.z;
    const int c = nc % C;
    const int n = nc / C;

    constexpr int INPUT_FRAME_H = TILES_H_PER_BLOCK * 4 + 2;
    constexpr int INPUT_FRAME_W = TILES_W_PER_BLOCK * 4 + 2;

    __shared__ float s_d[INPUT_FRAME_H][INPUT_FRAME_W];

    for (int i = threadIdx.x; i < INPUT_FRAME_H * INPUT_FRAME_W; i += BLOCK_SIZE)
    {
        const int s_col_idx = i % INPUT_FRAME_W;
        const int s_row_idx = i / INPUT_FRAME_W;

        const int row_idx = tile_row_idx_start * 4 + s_row_idx;
        const int col_idx = tile_col_idx_start * 4 + s_col_idx;

        if (row_idx >= 0 && row_idx < H && col_idx >= 0 && col_idx < W)
            s_d[s_row_idx][s_col_idx] = d[((n * C + c) * H + row_idx) * W + col_idx];
        else
            s_d[s_row_idx][s_col_idx] = 0;
    }
    __syncthreads();

    //BTd
    __shared__ float BTd[TILES_H_PER_BLOCK * TILES_W_PER_BLOCK][6][6];

    for (int i = threadIdx.x; i < TILES_H_PER_BLOCK * TILES_W_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        //col-wise d
        const int col_idx = i % 6;
        const int tile_idx = i / 6;

        const int tile_col_idx = tile_idx % TILES_W_PER_BLOCK;
        const int tile_row_idx = tile_idx / TILES_W_PER_BLOCK;

        if (tile_row_idx_start + tile_row_idx < TILES_H && tile_col_idx_start + tile_col_idx < TILES_W)
        {
            multiply_BT(s_d[tile_row_idx * 4 + 0][tile_col_idx * 4 + col_idx],s_d[tile_row_idx * 4 + 1][tile_col_idx * 4 + col_idx],s_d[tile_row_idx * 4 + 2][tile_col_idx * 4 + col_idx]
                ,s_d[tile_row_idx * 4 + 3][tile_col_idx * 4 + col_idx],s_d[tile_row_idx * 4 + 4][tile_col_idx * 4 + col_idx],s_d[tile_row_idx * 4 + 5][tile_col_idx * 4 + col_idx],
                BTd[tile_idx][0][col_idx],BTd[tile_idx][1][col_idx],BTd[tile_idx][2][col_idx],BTd[tile_idx][3][col_idx],BTd[tile_idx][4][col_idx],BTd[tile_idx][5][col_idx]);
        }
    }
    __syncthreads();

    __shared__ float BTdB[TILES_H_PER_BLOCK * TILES_W_PER_BLOCK][36];

    for (int i = threadIdx.x; i < TILES_H_PER_BLOCK * TILES_W_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        // row-wise BTd
        const int row_idx = i % 6;
        const int tile_idx = i / 6;

        const int tile_col_idx = tile_idx % TILES_W_PER_BLOCK;
        const int tile_row_idx = tile_idx / TILES_W_PER_BLOCK;

        if (tile_col_idx_start + tile_col_idx < TILES_W && tile_row_idx_start + tile_row_idx < TILES_H)
        {
            float in_row[6];
            for (int col_idx = 0; col_idx < 6; ++col_idx)
                in_row[col_idx] = BTd[tile_idx][row_idx][col_idx];

            float out_row[6];
            multiply_BT(BTd[tile_idx][row_idx][0],BTd[tile_idx][row_idx][1],BTd[tile_idx][row_idx][2],BTd[tile_idx][row_idx][3],BTd[tile_idx][row_idx][4],BTd[tile_idx][row_idx][5],
                BTdB[tile_idx][row_idx * 6 + 0],BTdB[tile_idx][row_idx * 6 + 1],BTdB[tile_idx][row_idx * 6 + 2],BTdB[tile_idx][row_idx * 6 + 3],BTdB[tile_idx][row_idx * 6 + 4],BTdB[tile_idx][row_idx * 6 + 5]);

            for (int col_idx = 0; col_idx < 6; ++col_idx)
                BTdB[tile_idx][row_idx * 6 + col_idx] = out_row[col_idx];
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 6 * 6 * TILES_H_PER_BLOCK * TILES_W_PER_BLOCK; i += BLOCK_SIZE)
    {
        const int tile_idx = i % (TILES_H_PER_BLOCK * TILES_W_PER_BLOCK);
        const int offset = i / (TILES_H_PER_BLOCK * TILES_W_PER_BLOCK);

        const int tile_col_idx = tile_idx % TILES_W_PER_BLOCK;
        const int tile_row_idx = tile_idx / TILES_W_PER_BLOCK;

        const int global_tile_col_idx = tile_col_idx + tile_col_idx_start;
        const int global_tile_row_idx = tile_row_idx + tile_row_idx_start;
        if (global_tile_col_idx < TILES_W && global_tile_row_idx < TILES_H)
            image_transform[(((offset * C + c) * N + n) * TILES_H + global_tile_row_idx) * TILES_W + global_tile_col_idx] = BTdB[tile_idx][offset];
    }
}
__device__ __forceinline__ void multiply_AT(const float in0,const float in1,const float in2,const float in3,const float in4,const float in5,
    float& out0,float& out1,float& out2,float& out3)
{
    //  A = {
    //     1,  1,  1,  1,  1,  0,
    //     0,  1, -1,  2, -2,  0,
    //     0,  1,  1,  4,  4,  0,
    //     0,  1, -1,  8, -8,  1
    // };
    float temp[4]={in1+in2,in1-in2,in3 + in4,in3 - in4};

    out0 = in0 + temp[0] + temp[2];
    out1 = temp[1] + 2 * temp[3];
    out2 = temp[0] + 4 * temp[2];
    out3 = temp[1] + 8 * temp[3] + in5;
}

template <int NUM_TILES_PER_BLOCK, int BLOCK_SIZE>
__global__ void winograd_ATtA(
    const float *t, const int N, const int K, const int TILES_H, const int TILES_W,
    float *out, const int OUT_H, const int OUT_W)//output_transform
{

    const int NUM_TILES = TILES_H * TILES_W;

    const int tile_idx_start = blockIdx.x * NUM_TILES_PER_BLOCK;
    const int k = blockIdx.y;
    const int n = blockIdx.z;

    __shared__ float shared_6x6[NUM_TILES_PER_BLOCK][6 * 6];
    #pragma omp parallel for
    for (int i = threadIdx.x; i < 6 * 6 * NUM_TILES_PER_BLOCK; i += BLOCK_SIZE)
    {
        const int local_tile_idx = i % NUM_TILES_PER_BLOCK;
        const int global_tile_idx = local_tile_idx + tile_idx_start;

        const int offset = i / NUM_TILES_PER_BLOCK;

        if (global_tile_idx < NUM_TILES)
            shared_6x6[local_tile_idx][offset] = t[((offset * K + k) * N + n) * NUM_TILES + global_tile_idx];
    }
    __syncthreads();

    //ATt
    __shared__ float shared_4x6[NUM_TILES_PER_BLOCK][4][6];
    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 6; i += BLOCK_SIZE)
    {
        //col-wise t
        const int col_idx = i % 6;
        const int tile_idx = i / 6;

        if (tile_idx_start + tile_idx < NUM_TILES)
        {

            multiply_AT(shared_6x6[tile_idx][0 * 6 + col_idx],shared_6x6[tile_idx][1 * 6 + col_idx],shared_6x6[tile_idx][2 * 6 + col_idx],shared_6x6[tile_idx][3 * 6 + col_idx],shared_6x6[tile_idx][4 * 6 + col_idx],shared_6x6[tile_idx][5 * 6 + col_idx],
                shared_4x6[tile_idx][0][col_idx],shared_4x6[tile_idx][1][col_idx],shared_4x6[tile_idx][2][col_idx],shared_4x6[tile_idx][3][col_idx]);

        }
    }
    __syncthreads();

    //ATtA
    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 4; i += BLOCK_SIZE)
    {
        //row-wise ATt
        const int row_idx = i % 4;
        const int tile_idx = i / 4;

        if (tile_idx + tile_idx_start < NUM_TILES)
        {
            multiply_AT(shared_4x6[tile_idx][row_idx][0], shared_4x6[tile_idx][row_idx][1],shared_4x6[tile_idx][row_idx][2],shared_4x6[tile_idx][row_idx][3],shared_4x6[tile_idx][row_idx][4],shared_4x6[tile_idx][row_idx][5],
                shared_6x6[tile_idx][row_idx * 4 + 0],shared_6x6[tile_idx][row_idx * 4 + 1],shared_6x6[tile_idx][row_idx * 4 + 2],shared_6x6[tile_idx][row_idx * 4 + 3]);
        }
    }
    __syncthreads();
    #pragma omp parallel for
    for (int i = threadIdx.x; i < NUM_TILES_PER_BLOCK * 4 * 4; i += BLOCK_SIZE)
    {
        const int tile_offset = i % 16;
        const int within_tile_col_idx = tile_offset % 4;
        const int within_tile_row_idx = tile_offset / 4;

        const int local_tile_idx = i / 16;
        const int global_tile_idx = tile_idx_start + local_tile_idx;

        const int global_tile_col_idx = global_tile_idx % TILES_W;
        const int global_tile_row_idx = global_tile_idx / TILES_W;

        const int map_x = global_tile_col_idx * 4 + within_tile_col_idx;
        const int map_y = global_tile_row_idx * 4 + within_tile_row_idx;

        if (global_tile_idx < NUM_TILES && map_x < OUT_W && map_y < OUT_H)
            out[((n * K + k) * OUT_H + map_y) * OUT_W + map_x] = shared_6x6[local_tile_idx][tile_offset];
    }
}

void winograd_convolution_cuda(float *h_img, const int N, const int C, const int H, const int W, const float *h_f, const int K, float *out)
{

    auto divUp = [](int x, int y)
    { return (x + y - 1) / y; };

    float *d_filter_transform;
    float *d_F;
    //filter transform
    {
        CUDA_CALL(cudaMalloc((void **)&d_F, K * C * 3 * 3 * sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_F, h_f, K * C * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CALL(cudaMalloc((void **)&d_filter_transform, 6 * 6 * K * C * sizeof(float)));

        const int NUM_KERNELS_PER_BLOCK = 4,
                  BLOCK_SIZE = 256;

        dim3 grid(divUp(K * C, 
                        NUM_KERNELS_PER_BLOCK));

        winograd_GgGT<NUM_KERNELS_PER_BLOCK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_F, K, C, d_filter_transform);
    }

    // input transform
    const int TILES_Y = divUp(H, 4);
    const int TILES_X = divUp(W, 4);

    float *d_image_transform;
    float *d_img;

    {
        CUDA_CALL(cudaMalloc((void **)&d_image_transform, 6 * 6 * C * N * TILES_Y * TILES_X * sizeof(float)));
        
        CUDA_CALL(cudaMalloc((void **)&d_img, N * C * H * W * sizeof(float)));
        CUDA_CALL(cudaMemcpy(d_img, h_img, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));

        const int TILES_Y_PER_BLOCK = 4, 
                  TILES_X_PER_BLOCK = 4,
                  BLOCK_SIZE = 256;

        dim3 grid(divUp(TILES_X, TILES_X_PER_BLOCK),
                  divUp(TILES_Y, TILES_Y_PER_BLOCK),
                  N * C
                );

        winograd_BTdB<TILES_Y_PER_BLOCK, TILES_X_PER_BLOCK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_img, N, C, H, W, d_image_transform, TILES_Y, TILES_X);
    }

    // U V
    float *d_M;
    {
        CUDA_CALL(cudaMalloc((void **)&d_M, 6 * 6 * K * N * TILES_Y * TILES_X * sizeof(float)));
        cublasHandle_t cbls_handle1;
        cublasCreate(&cbls_handle1);
        // cublasHandle_t cbls_handle2;
        // cublasCreate(&cbls_handle2);
        // cublasHandle_t cbls_handle3;
        // cublasCreate(&cbls_handle3);
        // cublasHandle_t cbls_handle4;
        // cublasCreate(&cbls_handle4);

        const float alpha = 1.0f,
                    beta = 0.0f;

        CUDA_CALL(cudaDeviceSynchronize()); // make sure filter and input transforms are ready
        CUDA_CALL(cudaFree(d_img));
        CUDA_CALL(cudaFree(d_F));
        CUBLAS_CALL(cublasSgemmStridedBatched(cbls_handle1,
                                              CUBLAS_OP_N, CUBLAS_OP_N,
                                              (N * TILES_Y * TILES_X), K, C,
                                              &alpha,
                                              d_image_transform, (N * TILES_Y * TILES_X), (C * N * TILES_Y * TILES_X),
                                              d_filter_transform, C, (K * C),
                                              &beta,
                                              d_M, (N * TILES_Y * TILES_X), (K * N * TILES_Y * TILES_X),
                                              36));
        // CUBLAS_CALL(cublasSgemmStridedBatched(cbls_handle2,
        // CUBLAS_OP_N, CUBLAS_OP_N,
        // (N * TILES_Y * TILES_X), K, C,
        // &alpha,
        // d_image_transform+9*(C * N * TILES_Y * TILES_X), (N * TILES_Y * TILES_X), (C * N * TILES_Y * TILES_X),
        // d_filter_transform+9*(K * C), C, (K * C),
        // &beta,
        // d_M+9*(K * N * TILES_Y * TILES_X), (N * TILES_Y * TILES_X), (K * N * TILES_Y * TILES_X),
        // 9));
        // CUBLAS_CALL(cublasSgemmStridedBatched(cbls_handle4,
        // CUBLAS_OP_N, CUBLAS_OP_N,
        // (N * TILES_Y * TILES_X), K, C,
        // &alpha,
        // d_image_transform+18*(C * N * TILES_Y * TILES_X), (N * TILES_Y * TILES_X), (C * N * TILES_Y * TILES_X),
        // d_filter_transform+18*(K * C), C, (K * C),
        // &beta,
        // d_M+27*(K * N * TILES_Y * TILES_X), (N * TILES_Y * TILES_X), (K * N * TILES_Y * TILES_X),
        // 9));
        // CUBLAS_CALL(cublasSgemmStridedBatched(cbls_handle4,
        // CUBLAS_OP_N, CUBLAS_OP_N,
        // (N * TILES_Y * TILES_X), K, C,
        // &alpha,
        // d_image_transform+27*(C * N * TILES_Y * TILES_X), (N * TILES_Y * TILES_X), (C * N * TILES_Y * TILES_X),
        // d_filter_transform+27*(K * C), C, (K * C),
        // &beta,
        // d_M+27*(K * N * TILES_Y * TILES_X), (N * TILES_Y * TILES_X), (K * N * TILES_Y * TILES_X),
        // 9));

        CUDA_CALL(cudaFree(d_filter_transform));
        CUDA_CALL(cudaFree(d_image_transform));
    }

    // inverse transform
    float *d_out;
    const int OUT_H = (H - 3 + 1),
              OUT_W = (W - 3 + 1);
    {
        CUDA_CALL(cudaMalloc((void **)&d_out, N * K * OUT_H * OUT_W * sizeof(float)));
        const int NUM_TILES_PER_BLOCK = 32,
                  BLOCK_SIZE = 256;

        dim3 grid(divUp(TILES_Y * TILES_X, NUM_TILES_PER_BLOCK), 
                  K, 
                  N
                );

        winograd_ATtA<NUM_TILES_PER_BLOCK, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(d_M, N, K, TILES_Y, TILES_X, d_out, OUT_H, OUT_W);

        CUDA_CALL(cudaFree(d_M));
    }
    CUDA_CALL(cudaDeviceSynchronize());

    float *h_out = (float *)malloc(N * K * OUT_H * OUT_W * sizeof(float));
    CUDA_CALL(cudaMemcpy(out, d_out, N * K * OUT_H * OUT_W * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_out));
}