#ifndef IMAGE_KMEANS_H
#define IMAGE_KMEANS_H

#include <cuda_runtime.h>
#include <npp.h>

Npp8u *imageKmeans(int k, Npp8u *h_pSrcImg_rgb, int nWidthPixels, int nHeightPixels, int nHostStep);

#endif