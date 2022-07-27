#include <image_copy.h>
#include <Exceptions.h>

std::tuple<Npp8u *, int> copyImageFromHostToDevice(Npp8u *h_pImg, int nWidthPixels, int nHeightPixels,
                                                   int nHostColorImgStep)
{
    if (nppGetStream() != 0)
    {
        nppSetStream(0);
    }

    int nDeviceColorImgStep;
    Npp8u *d_pImg = nppiMalloc_8u_C3(nWidthPixels, nHeightPixels, &nDeviceColorImgStep);
    NPP_ASSERT(d_pImg != NULL);

    cudaMemcpy2D(d_pImg, nDeviceColorImgStep, h_pImg, nHostColorImgStep,
                 3 * nWidthPixels, nHeightPixels, cudaMemcpyHostToDevice);

    return {d_pImg, nDeviceColorImgStep};
}

Npp8u *copyImageFromDeviceToHost(Npp8u *d_pImg, int nWidthPixels, int nHeightPixels,
                                 int nHostColorImgStep, int nDeviceColorImgStep)
{
    Npp8u *h_pImg = (Npp8u *)malloc(nHeightPixels * nHostColorImgStep);

    cudaMemcpy2D(h_pImg, nHostColorImgStep,
                 d_pImg, nDeviceColorImgStep,
                 3 * nWidthPixels, nHeightPixels, cudaMemcpyDeviceToHost);

    return h_pImg;
}
