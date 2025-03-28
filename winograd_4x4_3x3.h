#pragma once
void convWinograd_4x4_3x3(float *h_img, const int N, const int C, const int H, const int W, const float *h_f, const int K, float *out);