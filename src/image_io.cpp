#include <image_io.h>
#include <FreeImage.h>
#include <Exceptions.h>
#include <cstring>

// Error handler for FreeImage library.
//  In case this handler is invoked, it throws an NPP exception.
void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
    throw npp::Exception(zMessage);
}

Npp8u *loadColorImage(const std::string &sFilename, int *nWidthPixels, int *nHeightPixels,
                      int *pStepBytes)
{

    // set your own FreeImage error handler
    FreeImage_SetOutputMessage(FreeImageErrorHandler);

    FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());

    // no signature? try to guess the file format from the file extension
    if (eFormat == FIF_UNKNOWN)
    {
        eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
    }

    NPP_ASSERT(eFormat != FIF_UNKNOWN);

    // check that the plugin has reading capabilities ...
    FIBITMAP *pBitmap;

    if (FreeImage_FIFSupportsReading(eFormat))
    {
        pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
    }

    NPP_ASSERT(pBitmap != 0);
    NPP_ASSERT(FreeImage_GetImageType(pBitmap) == FIT_BITMAP);
    NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_RGB);
    NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 24);

    BYTE *bits = (BYTE *)FreeImage_GetBits(pBitmap);
    unsigned int width = FreeImage_GetWidth(pBitmap);
    unsigned int height = FreeImage_GetHeight(pBitmap);
    unsigned int pitch = FreeImage_GetPitch(pBitmap);

    Npp8u *h_pSrcImg = (Npp8u *)std::malloc(height * pitch);
    std::memcpy(h_pSrcImg, bits, height * pitch);

    *nWidthPixels = width;
    *nHeightPixels = height;
    *pStepBytes = pitch;

    return h_pSrcImg;
}

void saveColorImage(const std::string &rFileName, Npp8u *rImage, int nWidthPixels, int nHeightPixels, int nImgPitch)
{

    FIBITMAP *pResultBitmap = FreeImage_Allocate(nWidthPixels, nHeightPixels, 24);
    NPP_ASSERT_NOT_NULL(pResultBitmap);

    unsigned int nDstPitch = FreeImage_GetPitch(pResultBitmap);
    Npp8u *pDst = FreeImage_GetBits(pResultBitmap);

    NPP_ASSERT(nImgPitch == nDstPitch);
    std::memcpy(pDst, rImage, nDstPitch * nHeightPixels);

    // now save the result image
    bool bSuccess;
    bSuccess = FreeImage_Save(FIF_JPEG, pResultBitmap, rFileName.c_str(), 0) == TRUE;
    NPP_ASSERT_MSG(bSuccess, "Failed to save result image.");
}