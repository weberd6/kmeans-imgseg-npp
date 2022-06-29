#ifndef IMAGE_CONVERSION_H
#define IMAGE_CONVERSION_H

#include <cuda_runtime.h>
#include <npp.h>

Npp8u *convertRGBToHSV(Npp8u *d_pImg_rgb, int nWidthPixels, int nHeightPixels, int pStepBytes);
Npp8u *convertHSVToRGB(Npp8u *d_pImg_hsv, int nWidthPixels, int nHeightPixels, int pStepBytes);

#endif