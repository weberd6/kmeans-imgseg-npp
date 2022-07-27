#ifndef IMAGE_COPY_H
#define IMAGE_COPY_H

#include <npp.h>
#include <tuple>

std::tuple<Npp8u *, int> copyImageFromHostToDevice(Npp8u *h_pImg, int nWidthPixels, int nHeightPixels,
                                                   int nHostColorImgStep);

Npp8u *copyImageFromDeviceToHost(Npp8u *d_pImg, int nWidthPixels, int nHeightPixels,
                                 int nHostColorImgStep, int nDeviceColorImgStep);

#endif
