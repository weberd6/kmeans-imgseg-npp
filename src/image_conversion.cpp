
#include <image_conversion.h>
#include <helper_cuda.h>
#include <Exceptions.h>

Npp8u* convertRGBToHSV(Npp8u* d_pImg_rgb, int nWidthPixels, int nHeightPixels, int pStepBytes)
{
    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    Npp8u *d_pImg_hsv = NULL;
    cudaMalloc((Npp8u **)&d_pImg_hsv, nHeightPixels * pStepBytes);

    if (nppGetStream() != 0)
    {
        nppSetStream(0);
    }

    NPP_CHECK_NPP(nppiRGBToHSV_8u_C3R(d_pImg_rgb, pStepBytes,
                                      d_pImg_hsv, pStepBytes, fullSizeROI));

    return d_pImg_hsv;
}

Npp8u* convertHSVToRGB(Npp8u *d_pImg_hsv, int nWidthPixels, int nHeightPixels, int pStepBytes)
{
    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    Npp8u *d_pImg_rgb = NULL;
    cudaMalloc((Npp8u **)&d_pImg_rgb, nHeightPixels *pStepBytes);

    if (nppGetStream() != 0)
    {
        nppSetStream(0);
    }

    NPP_CHECK_NPP(nppiHSVToRGB_8u_C3R(d_pImg_hsv, pStepBytes,
                                      d_pImg_rgb, pStepBytes, fullSizeROI));

    return d_pImg_rgb;
}