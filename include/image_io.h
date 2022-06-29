#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <cuda_runtime.h>
#include <npp.h>
#include <string>

Npp8u *loadColorImage(const std::string &sFilename, int *nWidthPixels,
                      int *nHeightPixels, int *pStepBytes);

void saveColorImage(const std::string &rFileName, Npp8u *rImage,
                    int nWidthPixels, int nHeightPixels, int nImgStep);

#endif