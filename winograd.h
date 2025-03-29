#pragma once

void winograd_convolution(float *__restrict__ image,
                          const int irows,
                          const int icols,
                          const int C,
                          float *__restrict__ filter,
                          const int K,
                          const int batch,
                          float *__restrict__ out);
extern "C" void winograd_convolution_cuda(float *h_img, const int N, const int C, const int H, const int W, const float *h_f, const int K, float *out);