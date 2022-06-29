#include <Exceptions.h>

#include <string>
#include <fstream>
#include <iostream>
#include <cstring>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>
#include <image_io.h>
#include <image_kmeans.h>

bool printfNPPinfo(int argc, char *argv[]) {

    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
             libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
             (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
             (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

int main(int argc, char *argv[]) {

    printf("%s Starting...\n\n", argv[0]);

    try {

        std::string sFilename;
        char *filePath = NULL;
        int k = 6;

        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false) {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }

        if (filePath) {

            sFilename = filePath;

            // if we specify the filename at the command line, then we only test sFilename[0].
            int file_errors = 0;
            std::ifstream infile(sFilename.data(), std::ifstream::in);

            if (infile.good()) {
                std::cout << "Opened: <" << sFilename.data() << "> successfully!" << std::endl;
                file_errors = 0;
                infile.close();
            } else {
                std::cout << "Unable to open: <" << sFilename.data() << ">" << std::endl;
                file_errors++;
                infile.close();
            }

            if (file_errors > 0) {
                exit(EXIT_FAILURE);
            }

        } else {

            std::cout << "No input filename given" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos) {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_segmented.jpeg";

        if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output",
                                     &outputFilePath);
            sResultFilename = outputFilePath;
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "k")) {
            k = getCmdLineArgumentInt(argc, (const char **)argv, "k=");
        }

        std::cout << "k: " << k << std::endl;

        // Load image
        int nWidthPixels, nHeightPixels, nPitch;
        Npp8u *h_pSrcImg_rgb = loadColorImage(sFilename, &nWidthPixels, &nHeightPixels, &nPitch);

        Npp8u *h_pResImg_rgb = imageKmeans(k, h_pSrcImg_rgb, nWidthPixels, nHeightPixels, nPitch);

        // Save image
        saveColorImage(sResultFilename, h_pResImg_rgb, nWidthPixels, nHeightPixels, nPitch);

        free(h_pSrcImg_rgb);
        free(h_pResImg_rgb);

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {

        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
