//module;
#pragma once

#include <cudnn_frontend.h>
#include <cublas.h>
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>

#define checkCudaErr(...)                                                        \
    do {                                                                         \
        int64_t err = Helpers::checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        assert(err == 0);                                                       \
    } while (0)

#define checkCudnnErr(...)                                                        \
    do {                                                                          \
        int64_t err = Helpers::checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        assert(err == 0);                                                        \
    } while (0)

//export module LeNetHelpers;

import <assert.h>;
import <utility>;

namespace Helpers
{
    void bark();

//#define checkCudaErr(...)                                                        \
//    do {                                                                         \
//        int64_t err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
//        assert(err == 0);                                                       \
//    } while (0)
//
//#define checkCudnnErr(...)                                                        \
//    do {                                                                          \
//        int64_t err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
//        assert(err == 0);                                                        \
//    } while (0)

    int64_t checkCudaError(cudaError_t code, const char* expr, const char* file, int line);

    int64_t checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line);

    enum {
        X_TENSOR,
        Y_TENSOR,
        W_TENSOR,
        Z_TENSOR,
        B_TENSOR,
        AFTERADD_TENSOR,
        AFTERBIAS_TENSOR,
        AFTERCONV_TENSOR,
    };

    //enum class LENET_DESCRIPTORS : int
    //{
    //    X_TENSOR,
    //    Y_TENSOR,
    //    W_TENSOR,
    //    Z_TENSOR,
    //    B_TENSOR,
    //    AFTERADD_TENSOR,
    //    AFTERBIAS_TENSOR,
    //    AFTERCONV_TENSOR,
    //};

    using common_conv_descriptors =
        std::tuple<cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::ConvDesc>;

    using common_convbias_descriptors = std::tuple<
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor>;

    using lenet_descriptors = std::tuple<
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor,
        cudnn_frontend::Tensor>;

    bool
        allowAll(cudnnBackendDescriptor_t engine_config);

    bool
        isNonDeterministic(cudnnBackendDescriptor_t engine_config);

    //auto heurgen_method = [](cudnn_frontend::OperationGraph& opGraph) -> cudnn_frontend::EngineConfigList {
    //    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
    //        .setOperationGraph(opGraph)
    //        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
    //        .build();
    //    std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;

    //    auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    //    cudnn_frontend::EngineConfigList filtered_configs;
    //    cudnn_frontend::filter(engine_configs, filtered_configs, allowAll);
    //    return filtered_configs;
    //};

    //// Method for engine config generator based on fallback list
    //auto fallback_method = [](cudnn_frontend::OperationGraph& opGraph) -> cudnn_frontend::EngineConfigList {
    //    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
    //        .setOperationGraph(opGraph)
    //        .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
    //        .build();
    //    auto& fallback_list = fallback.getFallbackList();

    //    cudnn_frontend::EngineConfigList filtered_configs;
    //    // We create this filter to pre-remove configs being passed to cudnnFind.
    //    // This is just a sample and is not necessary
    //    cudnn_frontend::filter(fallback_list, filtered_configs, isNonDeterministic);

    //    return filtered_configs;
    //};

    void generateStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat);
    
    common_conv_descriptors
        create_common_descriptors(int64_t* x_dim,
            int64_t* padA,
            int64_t* convstrideA,
            int64_t* dilationA,
            int64_t* w_dim,
            int64_t* y_dim,
            cudnnDataType_t dataType,
            cudnnConvolutionMode_t mode);

    common_convbias_descriptors create_lenet_descriptors(
        int64_t* x_dim,
        int64_t* padA,
        int64_t* convstrideA,
        int64_t* dilationA,
        int64_t* w_dim,
        int64_t* y_dim,
        cudnnDataType_t dataType,
        cudnnDataType_t computeType);
    
    cudnn_frontend::OperationGraph create_operation_graph(
            common_conv_descriptors& descriptors,
            cudnnBackendDescriptorType_t mode,
            cudnnHandle_t handle_);

    cudnn_frontend::OperationGraph create_lenet_operation_graph(
        common_convbias_descriptors& descriptors,
        int64_t* pad,
        int64_t* convstride,
        int64_t* dilation,
        cudnnHandle_t handle_);

    cudnn_frontend::EngineConfigList generateConfigList(cudnn_frontend::OperationGraph& opGraph);

    int64_t getFwdConvDilatedFilterDim(int64_t filterDim, int64_t dilation);

    int64_t getFwdConvPaddedImageDim(int64_t tensorDim, int64_t pad);

    int64_t getFwdConvOutputDim(
        int64_t tensorDim,
        int64_t pad,
        int64_t filterDim,
        int64_t stride,
        int64_t dilation);

    void initFilter(float* hostW, size_t size);

    cudnn_frontend::ExecutionPlan get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_);

    void initImage(float* image, int64_t imageSize);
    void initImage(int8_t* image, int64_t imageSize);

    void initImagev2(float* image, int64_t imageSize);

    template <typename T_ELEM>
    struct Surface {
        T_ELEM* devPtr = NULL;
        T_ELEM* hostPtr = NULL;
        T_ELEM* hostRefPtr = NULL;
        int64_t n_elems = 0;

        explicit Surface(int64_t n_elems, bool hasRef = false) : n_elems(n_elems) {
            checkCudaErr(cudaMalloc((void**)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));
            hostPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostPtr[0]));
            if (hasRef) {
                hostRefPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostRefPtr[0]));
            }
            initImage(hostPtr, n_elems);
            checkCudaErr(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
            checkCudaErr(cudaDeviceSynchronize());
        }

        explicit Surface(int64_t n_elems, bool hasRef, bool isInterleaved) {
            (void)isInterleaved;
            checkCudaErr(cudaMalloc((void**)&(devPtr), (n_elems) * sizeof(devPtr[0])));
            hostPtr = (T_ELEM*)calloc(n_elems, sizeof(hostPtr[0]));
            if (hasRef) {
                hostRefPtr = (T_ELEM*)calloc(n_elems, sizeof(hostRefPtr[0]));
            }
            initImage(hostPtr, n_elems);
            uint32_t* temp = (uint32_t*)hostPtr;
            for (size_t i = 0; i < n_elems; i = i + 2) {
                temp[i + 1] = 1u;
            }

            checkCudaErr(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
            checkCudaErr(cudaDeviceSynchronize());
        }

        explicit Surface(int64_t size, bool hasRef, T_ELEM fillValue) : n_elems(size) {
            checkCudaErr(cudaMalloc((void**)&(devPtr), (size) * sizeof(devPtr[0])));
            hostPtr = (T_ELEM*)calloc(size, sizeof(hostPtr[0]));
            if (hasRef) {
                hostRefPtr = (T_ELEM*)calloc(n_elems, sizeof(hostRefPtr[0]));
            }
            for (int i = 0; i < size; i++) {
                hostPtr[i] = fillValue;
            }
            checkCudaErr(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
            checkCudaErr(cudaDeviceSynchronize());
        }

        ~Surface() {
            if (devPtr) {
                cudaFree(devPtr);
                devPtr = nullptr;
            }
            if (hostPtr) {
                free(hostPtr);
                hostPtr = nullptr;
            }
            if (hostRefPtr) {
                free(hostRefPtr);
                hostRefPtr = nullptr;
            }
        }

        void devToHostSync() {
            cudaDeviceSynchronize();
            checkCudaErr(cudaMemcpy(hostPtr, devPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
        }

        void hostToDevSync() {
            cudaDeviceSynchronize();
            checkCudaErr(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();
        }
    };
}