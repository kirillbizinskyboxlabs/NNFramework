//module;
//
//#include <cudnn_frontend.h>
//
//module LeNet;
//
//import <vector>;
//import <string>;
//
//using Words = std::vector<std::string>;
//
//namespace LeNet
//{
//	Words getWords()
//	{
//		return Words();
//	}
//}

//
//import <iostream>;
//
////import :Helpers;
//
//namespace LeNet
//{
//    void RigidTest()
//    {
//        std::cout << "Rigid LeNet Test v0.1" << std::endl;
//        //INFO("TEST_CASE :: Use heuristics for engine generation");
//        int64_t dimA[] = { 1, 1, 28, 28 };
//        int64_t filterdimA[] = { 1, 6, 5, 5 };
//        int64_t outdimA[] = { 0, 0, 0, 0 }; // Computed Below
//        int64_t padA[] = { 2, 2 };
//        int64_t dilationA[] = { 1, 1 };
//        int64_t convstrideA[] = { 1, 1 };
//
//        int numErrors = 0;
//
//        //outdimA[0] = dimA[0];
//        //outdimA[1] = filterdimA[0];
//        //for (int dim = 0; dim < 2; dim++) {
//        //    outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
//        //}
//
//
//        //cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;
//
//        //printf("====DIMENSIONS====\n");
//        //printf("input dims are %lld, %lld, %lld, %lld\n", dimA[0], dimA[1], dimA[2], dimA[3]);
//        //printf("filter dims are %lld, %lld, %lld, %lld\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
//        //printf("output dims are %lld, %lld, %lld, %lld\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);
//
//
//        //int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
//        //int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
//        //int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];
//
//        //float* devPtrX = NULL;
//        //float* devPtrW = NULL;
//        //float* devPtrY = NULL;
//
//        //checkCudaErr(cudaMalloc((void**)&(devPtrX), size_t((Xsize) * sizeof(devPtrX[0]))));
//        //checkCudaErr(cudaMalloc((void**)&(devPtrW), size_t((Wsize) * sizeof(devPtrW[0]))));
//        //checkCudaErr(cudaMalloc((void**)&(devPtrY), size_t((Ysize) * sizeof(devPtrY[0]))));
//    }
//}