#include <image_kmeans.h>
#include <image_conversion.h>
#include <helper_cuda.h>
#include <Exceptions.h>
#include <tuple>
#include <iostream>

std::tuple<Npp8u *, int> copyImageFromHostToDevice(Npp8u *h_pImg, int nWidthPixels, int nHeightPixels, int nHostColorImgStep)
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

Npp8u *copyImageFromDeviceToHost(Npp8u *d_pImg, int nWidthPixels, int nHeightPixels, int nHostColorImgStep, int nDeviceColorImgStep) {

    Npp8u *h_pImg = (Npp8u *)malloc(nHeightPixels * nHostColorImgStep);

    cudaMemcpy2D(h_pImg, nHostColorImgStep,
                 d_pImg, nDeviceColorImgStep,
                 3 * nWidthPixels, nHeightPixels, cudaMemcpyDeviceToHost);

    return h_pImg;
}

Npp8u *createAndInitializeCentroids(int k, Npp8u *d_pSrcImg_hsv, int nWidthPixels, int nHeightPixels,
                                    int nDeviceColorImgStep, int nHostColorImgStep)
{
    // Determine initial centroids randomly

    Npp8u *d_pCentroids = NULL;
    cudaMalloc((Npp8u **)&d_pCentroids, k * nHeightPixels * nDeviceColorImgStep);

    Npp8u *h_pSrcImg_hsv = (Npp8u *)malloc(nHeightPixels * nHostColorImgStep);
    cudaMemcpy2D(h_pSrcImg_hsv, nHostColorImgStep, d_pSrcImg_hsv, nDeviceColorImgStep,
                 3 * nWidthPixels, nHeightPixels, cudaMemcpyDeviceToHost);

    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    if (nppGetStream() != 0)
    {
        nppSetStream(0);
    }

    for (int i = 0; i < k; i++)
    {
        int x = rand() % nWidthPixels;
        int y = rand() % nHeightPixels;

        int img_idx = 3 * (y * nWidthPixels + x);

        std::cout << "Centroid " << i << ": ("
                  << +h_pSrcImg_hsv[img_idx] << ", "
                  << +h_pSrcImg_hsv[img_idx + 1] << ", "
                  << +h_pSrcImg_hsv[img_idx + 2] << ")" << std::endl;

        Npp8u *d_pCentroid = &d_pCentroids[i * nHeightPixels * nDeviceColorImgStep];

        NPP_CHECK_NPP(nppiSet_8u_C3R(&h_pSrcImg_hsv[img_idx], d_pCentroid, nDeviceColorImgStep, fullSizeROI));
    }

    free(h_pSrcImg_hsv);

    return d_pCentroids;
}

std::tuple<cudaStream_t *, NppStreamContext *> createStreamContexts(int k)
{

    cudaStream_t *streams = (cudaStream_t *)malloc(k * sizeof(cudaStream_t));
    NppStreamContext *nppStreamContexts = (NppStreamContext *)malloc(k * sizeof(NppStreamContext));
    cudaDeviceProp oDeviceProperties;

    for (int i = 0; i < k; i++)
    {
        cudaStreamCreate(&streams[i]);
        nppStreamContexts[i].hStream = streams[i];

        cudaGetDevice(&nppStreamContexts[i].nCudaDeviceId);

        cudaGetDeviceProperties(&oDeviceProperties, nppStreamContexts[i].nCudaDeviceId);
        nppStreamContexts[i].nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
        nppStreamContexts[i].nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
        nppStreamContexts[i].nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
        nppStreamContexts[i].nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;

        cudaDeviceGetAttribute(&nppStreamContexts[i].nCudaDevAttrComputeCapabilityMajor,
                               cudaDevAttrComputeCapabilityMajor,
                               nppStreamContexts[i].nCudaDeviceId);

        cudaDeviceGetAttribute(&nppStreamContexts[i].nCudaDevAttrComputeCapabilityMinor,
                               cudaDevAttrComputeCapabilityMinor,
                               nppStreamContexts[i].nCudaDeviceId);

        cudaStreamGetFlags(nppStreamContexts[i].hStream, &nppStreamContexts[i].nStreamFlags);
    }

    return {streams, nppStreamContexts};
}

std::tuple<Npp8u *, Npp8u *, Npp8u *, Npp8u *, int> allocateSingleChannelImages(int k, int nWidthPixels, int nHeightPixels) {

    if (nppGetStream() != 0)
    {
        nppSetStream(0);
    }

    int nSingleChannelStep;
    Npp8u *d_pAssignedCentroids = nppiMalloc_8u_C1(nWidthPixels, nHeightPixels, &nSingleChannelStep);
    NPP_ASSERT(d_pAssignedCentroids != NULL);

    Npp8u *d_pDistances = NULL;
    cudaMalloc((Npp8u **)&d_pDistances, k * nHeightPixels * nSingleChannelStep);
    NPP_ASSERT(d_pDistances != NULL);

    Npp8u *d_pMinDistance = NULL;
    cudaMalloc((Npp8u **)&d_pMinDistance, nHeightPixels * nSingleChannelStep);
    NPP_ASSERT(d_pMinDistance != NULL);

    Npp8u *d_pMasks = NULL;
    cudaMalloc((Npp8u **)&d_pMasks, k * nHeightPixels * nSingleChannelStep);
    NPP_ASSERT(d_pMasks != NULL);

    return {d_pAssignedCentroids, d_pDistances, d_pMinDistance, d_pMasks, nSingleChannelStep};
}

std::tuple<Npp8u *, int> allocateMeanBuffers(int k, int nWidthPixels, int nHeightPixels) {

    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    if (nppGetStream() != 0)
    {
        nppSetStream(0);
    }

    int nBufferSize;
    NPP_CHECK_NPP(nppiMeanGetBufferHostSize_8u_C3CMR(fullSizeROI, &nBufferSize));

    Npp8u *d_pMeanBuffers = NULL;
    cudaMalloc((Npp8u **)&d_pMeanBuffers, k*nBufferSize);
    NPP_ASSERT(d_pMeanBuffers != NULL);

    return {d_pMeanBuffers, nBufferSize};
}

void HSVDistanceBetweenPixels(const Npp8u *d_pImg1, const Npp8u *d_pImg2, int nColorImgStep,
                              Npp8u *d_pDistance, int nDistanceStep,
                              int nWidthPixels, int nHeightPixels, NppStreamContext ctx)
{
    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    Npp8u *d_pDifference = NULL;
    cudaMallocAsync((Npp8u **)&d_pDifference, nHeightPixels * nColorImgStep, ctx.hStream);
    NPP_ASSERT(d_pDifference != NULL);

    Npp8u *d_pDistance255 = NULL;
    cudaMallocAsync((Npp8u **)&d_pDistance255, nHeightPixels * nDistanceStep, ctx.hStream);
    NPP_ASSERT(d_pDistance255 != NULL);

    // Compute img absolute difference (abs(h1-h0),abs(s1-s0),abs(v1-v0))
    NPP_CHECK_NPP(nppiAbsDiff_8u_C3R_Ctx(d_pImg1, nColorImgStep,
                                         d_pImg2, nColorImgStep,
                                         d_pDifference, nColorImgStep,
                                         fullSizeROI, ctx));

    // Extract abs(h1-h0) channel
    NPP_CHECK_NPP(nppiCopy_8u_C3C1R_Ctx(d_pDifference, nColorImgStep,
                                        d_pDistance, nDistanceStep,
                                        fullSizeROI, ctx));

    // 255 - abs(h1-h0)
    NPP_CHECK_NPP(nppiAbsDiffC_8u_C1R_Ctx(d_pDistance, nDistanceStep,
                                          d_pDistance255, nDistanceStep,
                                          fullSizeROI, 255, ctx));

    // min(abs(h1-h0), 255-abs(h1-h0))
    NPP_CHECK_NPP(nppiMinEvery_8u_C1IR_Ctx(d_pDistance255, nDistanceStep,
                                           d_pDistance, nDistanceStep,
                                           fullSizeROI, ctx));

    // Rescale h difference after taking into account circular nature of h channel
    NPP_CHECK_NPP(nppiMulC_8u_C1IRSfs_Ctx(2, d_pDistance, nDistanceStep, fullSizeROI, 0, ctx));

    // Copy back h difference
    NPP_CHECK_NPP(nppiCopy_8u_C1C3R_Ctx(d_pDistance, nDistanceStep,
                                        d_pDifference, nColorImgStep,
                                        fullSizeROI, ctx));

    // Squared difference in place
    NPP_CHECK_NPP(nppiSqr_8u_C3IRSfs_Ctx(d_pDifference, nColorImgStep, fullSizeROI, 8, ctx));

    // Sum of square differences
    Npp32f aCoeffs[3] = {0.333, 0.333, 0.333};
    NPP_CHECK_NPP(nppiColorToGray_8u_C3C1R_Ctx(d_pDifference, nColorImgStep,
                                               d_pDistance, nDistanceStep,
                                               fullSizeROI, aCoeffs, ctx));

    // Sqrt of the sum of square differences
    NPP_CHECK_NPP(nppiSqrt_8u_C1IRSfs_Ctx(d_pDistance, nDistanceStep,
                                          fullSizeROI, 0, ctx));

    // Rescale
    NPP_CHECK_NPP(nppiMulC_8u_C1IRSfs_Ctx(16, d_pDistance, nDistanceStep, fullSizeROI, 0, ctx));

    cudaFreeAsync(d_pDifference, ctx.hStream);
    cudaFreeAsync(d_pDistance255, ctx.hStream);
}

void calculateDistances(int k, const Npp8u *d_pSrcImg, const Npp8u *d_pCentroids, int nColorImgStep,
                        Npp8u *d_pDistances, int nDistanceStep,
                        int nWidthPixels, int nHeightPixels, NppStreamContext *streamContexts)
{
    // For each pixel calculate distances between each centroid
    for (int j = 0; j < k; j++)
    {
        const Npp8u *d_pCentroid = &d_pCentroids[j * nHeightPixels * nColorImgStep];
        Npp8u *d_pDistance = &d_pDistances[j * nHeightPixels * nDistanceStep];

        HSVDistanceBetweenPixels(d_pSrcImg, d_pCentroid, nColorImgStep,
                                 d_pDistance, nDistanceStep,
                                 nWidthPixels, nHeightPixels, streamContexts[j]);
    }
}

void assignmentStep(int k, const Npp8u *d_pSrcImg, const Npp8u *d_pCentroids, int nColorImgStep,
                    Npp8u *d_pDistances, Npp8u *d_pMinDistance, int nDistanceStep,
                    Npp8u *d_pAssignedCentroids, Npp8u *d_pMask, int nMaskStep,
                    int nWidthPixels, int nHeightPixels, NppStreamContext *streamContexts)
{
    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    calculateDistances(k, d_pSrcImg, d_pCentroids, nColorImgStep,
                       d_pDistances, nDistanceStep,
                       nWidthPixels, nHeightPixels, streamContexts);

    if (nppGetStream() != 0) {
        nppSetStream(0);
    }

    // Initialize minimum distances to maximum value
    NPP_CHECK_NPP(nppiSet_8u_C1R(255, d_pMinDistance, nDistanceStep, fullSizeROI));

    // For each pixel find closest centroid and assign it
    for (int j = 0; j < k; j++)
    {
        Npp8u *d_pDistance = &d_pDistances[j * nHeightPixels * nDistanceStep];

        // Compare distance with min_distance
        NPP_CHECK_NPP(nppiCompare_8u_C1R(d_pDistance, nDistanceStep,
                                         d_pMinDistance, nDistanceStep,
                                         d_pMask, nMaskStep, fullSizeROI, NPP_CMP_LESS));

        // Mask copy distance to min_distance
        NPP_CHECK_NPP(nppiCopy_8u_C1MR(d_pDistance, nDistanceStep,
                                       d_pMinDistance, nDistanceStep, fullSizeROI,
                                       d_pMask, nMaskStep));

        // Mask set assigned centroid
        NPP_CHECK_NPP(nppiSet_8u_C1MR(j,
                                      d_pAssignedCentroids, nMaskStep, fullSizeROI,
                                      d_pMask, nMaskStep));
    }
}

Npp64f *recomputeCentroids(int k, const Npp8u *d_pSrcImg, Npp8u *d_pCentroids, int nColorImgStep,
                           const Npp8u *d_pAssignedCentroids, Npp8u *d_pMasks, int nMaskStep,
                           Npp8u *d_pMeanBuffers, int nMeanBufferSize,
                           int nWidthPixels, int nHeightPixels, NppStreamContext *streamContexts)
{
    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    Npp64f *d_pAvgValues = NULL;
    cudaMalloc((Npp64f **)&d_pAvgValues, k * 3 * sizeof(Npp64f));
    NPP_ASSERT(d_pAvgValues != NULL);

    Npp64f *h_pAvgValues = (Npp64f *)malloc(k * 3 * sizeof(Npp64f));
    NPP_ASSERT(h_pAvgValues != NULL);

    for (int j = 0; j < k; j++)
    {
        Npp8u *d_pCentroid = &d_pCentroids[j * nHeightPixels * nColorImgStep];
        Npp8u *d_pMask = &d_pMasks[j * nHeightPixels * nMaskStep];
        Npp64f *d_pAvgValue = &d_pAvgValues[j * 3];
        Npp8u *d_pMeanBuffer = &d_pMeanBuffers[j * nMeanBufferSize];
        Npp8u centroid = j;

        // Get mask for assigned_centroid == centroid
        NPP_CHECK_NPP(nppiCompareC_8u_C1R_Ctx(d_pAssignedCentroids, nMaskStep, centroid,
                                              d_pMask, nMaskStep,
                                              fullSizeROI, NPP_CMP_EQ, streamContexts[j]));

        // H average
        NPP_CHECK_NPP(nppiMean_8u_C3CMR_Ctx(d_pSrcImg, nColorImgStep,
                                            d_pMask, nMaskStep,
                                            fullSizeROI, 1, d_pMeanBuffer,
                                            &d_pAvgValue[0], streamContexts[j]));

        // S average
        NPP_CHECK_NPP(nppiMean_8u_C3CMR_Ctx(d_pSrcImg, nColorImgStep,
                                            d_pMask, nMaskStep,
                                            fullSizeROI, 2, d_pMeanBuffer,
                                            &d_pAvgValue[1], streamContexts[j]));

        // V average
        NPP_CHECK_NPP(nppiMean_8u_C3CMR_Ctx(d_pSrcImg, nColorImgStep,
                                            d_pMask, nMaskStep,
                                            fullSizeROI, 3, d_pMeanBuffer,
                                            &d_pAvgValue[2], streamContexts[j]));
    }

    cudaMemcpy(h_pAvgValues, d_pAvgValues, 3 * k * sizeof(Npp64f), cudaMemcpyDeviceToHost);
    cudaFree(d_pAvgValues);

    return h_pAvgValues;
}

void updateStep(int k, const Npp8u *d_pSrcImg, Npp8u *d_pCentroids, int nColorImgStep,
                Npp8u *d_pAssignedCentroids, Npp8u *d_pMasks, int nMaskStep,
                Npp8u *d_pMeanBuffers, int nMeanBufferSize, int nWidthPixels, int nHeightPixels,
                NppStreamContext *streamContexts)
{

    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    Npp64f *h_pAvgValues = recomputeCentroids(k, d_pSrcImg, d_pCentroids, nColorImgStep,
                                              d_pAssignedCentroids, d_pMasks, nMaskStep,
                                              d_pMeanBuffers, nMeanBufferSize,
                                              nWidthPixels, nHeightPixels, streamContexts);
    // Update centroids

    for (int j = 0; j < k; j++)
    {
        Npp8u *d_pCentroid = &d_pCentroids[j * nHeightPixels * nColorImgStep];
        Npp64f *h_pCentroidValues = &h_pAvgValues[j * 3];
        Npp8u centroidValues[3] = {(Npp8u)h_pCentroidValues[0],
                                   (Npp8u)h_pCentroidValues[1],
                                   (Npp8u)h_pCentroidValues[2]};

        NPP_CHECK_NPP(nppiSet_8u_C3R_Ctx(centroidValues, d_pCentroid, nColorImgStep, fullSizeROI, streamContexts[j]));
    }

    free(h_pAvgValues);
}

Npp8u *copyAssignedCentroidsToImage(int k, const Npp8u *d_pAssignedCentroids, Npp8u *d_pMask, int nMaskStep,
                                    const Npp8u *d_pCentroids, int nColorImgStep,
                                    int nWidthPixels, int nHeightPixels, NppStreamContext *streamContexts)
{
    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    Npp8u *d_pResImg_hsv = NULL;
    cudaMalloc((Npp8u **)&d_pResImg_hsv, nHeightPixels * nColorImgStep);

    for (int i = 0; i < k; i++)
    {
        const Npp8u *d_pCentroid = &d_pCentroids[i * nHeightPixels * nColorImgStep];

        NPP_CHECK_NPP(nppiCompareC_8u_C1R_Ctx(d_pAssignedCentroids, nMaskStep, i,
                                              d_pMask, nMaskStep, fullSizeROI, NPP_CMP_EQ,
                                              streamContexts[i]));

        // Mask copy centroids into result image
        NPP_CHECK_NPP(nppiCopy_8u_C3MR_Ctx(d_pCentroid, nColorImgStep,
                                           d_pResImg_hsv, nColorImgStep, fullSizeROI,
                                           d_pMask, nMaskStep, streamContexts[i]));
    }

    return d_pResImg_hsv;
}

Npp8u *imageKmeans(int k, Npp8u *h_pSrcImg_rgb, int nWidthPixels, int nHeightPixels, int nHostColorImgStep)
{
    int nDeviceColorImgStep;
    int nSingleChannelStep;
    Npp8u *d_pSrcImg_rgb;
    Npp8u *d_pSrcImg_hsv;
    cudaStream_t *streams;
    NppStreamContext *nppStreamContexts;
    Npp8u *d_pCentroidImgs;
    Npp8u *d_pAssignedCentroids;
    Npp8u *d_pDistances;
    Npp8u *d_pMinDistance;
    Npp8u *d_pMasks;
    Npp8u *d_pMeanBuffers;
    int nMeanBufferSize;

    NppiSize fullSizeROI = {(int)nWidthPixels, (int)nHeightPixels};

    std::tie(d_pSrcImg_rgb, nDeviceColorImgStep) =
        copyImageFromHostToDevice(h_pSrcImg_rgb, nWidthPixels, nHeightPixels, nHostColorImgStep);

    d_pSrcImg_hsv = convertRGBToHSV(d_pSrcImg_rgb, nWidthPixels, nHeightPixels, nDeviceColorImgStep);

    d_pCentroidImgs = createAndInitializeCentroids(k, d_pSrcImg_hsv, nWidthPixels, nHeightPixels,
                                                   nDeviceColorImgStep, nHostColorImgStep);

    std::tie(streams, nppStreamContexts) = createStreamContexts(k);

    std::tie(d_pAssignedCentroids,
             d_pDistances,
             d_pMinDistance,
             d_pMasks,
             nSingleChannelStep) = allocateSingleChannelImages(k, nWidthPixels, nHeightPixels);

    std::tie(d_pMeanBuffers, nMeanBufferSize) = allocateMeanBuffers(k, nWidthPixels, nHeightPixels);

    int max_iter = 10;
    for (int i = 0; i < max_iter; i++)
    {
        assignmentStep(k, d_pSrcImg_hsv, d_pCentroidImgs, nDeviceColorImgStep,
                       d_pDistances, d_pMinDistance, nSingleChannelStep,
                       d_pAssignedCentroids, d_pMasks, nSingleChannelStep,
                       nWidthPixels, nHeightPixels, nppStreamContexts);

        updateStep(k, d_pSrcImg_hsv, d_pCentroidImgs, nDeviceColorImgStep,
                   d_pAssignedCentroids, d_pMasks, nSingleChannelStep,
                   d_pMeanBuffers, nMeanBufferSize, nWidthPixels, nHeightPixels, nppStreamContexts);

        cudaDeviceSynchronize();
    }

    // Set pixel value as the mean of the centroid that its assigned to

    Npp8u *d_pResImg_hsv = copyAssignedCentroidsToImage(k, d_pAssignedCentroids, d_pMasks, nSingleChannelStep,
                                                        d_pCentroidImgs, nDeviceColorImgStep,
                                                        nWidthPixels, nHeightPixels, nppStreamContexts);

    // Convert from HSV to RGB
    Npp8u *d_pResImg_rgb = convertHSVToRGB(d_pResImg_hsv, nWidthPixels, nHeightPixels, nDeviceColorImgStep);

    Npp8u *h_pResImg_rgb = copyImageFromDeviceToHost(d_pResImg_rgb, nWidthPixels, nHeightPixels,
                                                     nHostColorImgStep, nDeviceColorImgStep);

    nppiFree(d_pSrcImg_rgb);
    cudaFree(d_pSrcImg_hsv);
    cudaFree(d_pCentroidImgs);

    nppiFree(d_pAssignedCentroids);
    cudaFree(d_pDistances);
    cudaFree(d_pMinDistance);
    cudaFree(d_pMasks);
    cudaFree(d_pMeanBuffers);
    cudaFree(d_pResImg_hsv);
    cudaFree(d_pResImg_rgb);

    for (int i = 0; i < k; i++)
    {
        cudaStreamDestroy(nppStreamContexts[i].hStream);
    }
    free(streams);
    free(nppStreamContexts);

    return h_pResImg_rgb;
}