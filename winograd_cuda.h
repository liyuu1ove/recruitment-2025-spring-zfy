#pragma once
void winograd_convolution_cuda(float *h_img, const int N, const int C, const int H, const int W, const float *h_f, const int K, float *out);