# K-means Image Segmentation
An implementation of k-means image segmentation for color images that uses NVIDIA Performance Primitives.

## Installation
1. Make sure CUDA_ROOT_DIR in the Makefile is set to the directory where CUDA is installed.
2. run make

## Usage
./kmeans_seg_npp -input *input-filename* [-output *output-filename*] [-k=*number-of-centroids*]

*input-filename* is the name of an RGB (24 bpp) colored image file (e.g. JPEG)
*output-filename* is the name of the file that the final segmented image will be written to
*number-of-centroids* is the number of colors the segmented image will use

## Description of K-means Image Segmentation
K-means is a data clustering algorithm that alternates between two main step, namely the assignment step and the update step. First, k centroids are determined randomly by randomly selecting a pixel in the image. Then in the assignment step, the closest centroid to each pixel is determined using a distance function and the closest centroid to each pixel is assigned to it. Then in the update step, each centroid is recalculated using the average of all the pixels that were assigned to it. The assignment and update step are repeated until the centroids remain mostly unchanged and the final segmented image is then created by setting each pixel to its assigned centroid color.

### Implementation details
#### Distance function
The distance used between pixels in this implementation is the L2 distance squared in the HSV color space with the difference in Hue adjusted to account for the circular nature of it. Given two pixels (H1, S1, V1) and (H2, S2, V2) the distance is calculated as 1/3 * 1/256 * ((2 * min(abs(H2-H1), 255-abs(H2-H1)))^2 + (abs(S2-S1))^2 + (abs(V2-V1))^2). The 1/3 and 1/256 scaling factors are used in intermediate calculations to ensure that values remain between 0 and 255.

#### CUDA specific
When centroids, distances, and other intermediate calculation are done, entire images are used so that parallel pixel computations can be done using CUDA kernels. To increase performance, memory is reused when possible to avoid too many calls to cudaMalloc and cudaFree. Also, a stream per centroid is also used when possible to allow asynchronous execution for independent calculations.
