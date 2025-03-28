#pragma once
void cublasMatrix(const int64_t M,const int64_t K,const int64_t N,float *hostA, float *hostB, float *hostC); 
void cuda_image_transform(float *__restrict__ packed_image,
    float *__restrict__ V,
    const V_shape_t vs,
    const tiling_info_t ti,
    const int64_t collapsed_dim_size);