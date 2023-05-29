#include "DevUtils.h"

#include <cudnn_frontend.h>

import <random>;
import <format>;
import <iostream>;

//anonymus namespace. Not too good TODO:rethink
namespace
{
    bool allowAll(cudnnBackendDescriptor_t engine_config) {
        (void)engine_config;
        return false;
    }

    bool isNonDeterministic(cudnnBackendDescriptor_t engine_config) {
        return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(engine_config);
    }
}

void Utils::initImage(float* image, int64_t imageSize)
{
    constexpr float numerator = 2.0f; // depending on optimization strategy different numerators are suggested. Althernative is to use 3.0f

    std::random_device dev;
    std::mt19937 gen(dev());
    float deviation = sqrt(numerator / imageSize);
    std::uniform_real_distribution<float> distribution(-deviation, deviation);

    for (int64_t i = 0; i < imageSize; ++i)
    {
        image[i] = distribution(gen);
    }
}

cudnn_frontend::ExecutionPlan Utils::get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_)
{
#if (CUDNN_VERSION >= 8200)
    {
        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
            .setOperationGraph(opGraph)
            .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
            .build();

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

        // Try engine configs returned by the heuristics and pick up the first one that works.
        for (auto& ecfg : engine_config) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle_)
                    .setEngineConfig(ecfg, opGraph.getTag())
                    .build();
                return plan;
            }
            catch (cudnn_frontend::cudnnException&) {
                continue;
            }
        }
    }
#endif

    {
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses =
            cudnn_frontend::get_heuristics_list<1>({
            "heuristics_fallback"
                }, opGraph, allowAll, filtered_configs, true);

        std::cout << "get_heuristics_list Statuses: ";
        for (auto status : statuses) {
            std::cout << cudnn_frontend::to_string(status) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
    }
}

namespace
{
    int64_t getFwdConvDilatedFilterDim(int64_t filterDim, int64_t dilation) {
        return ((filterDim - 1) * dilation) + 1;
    }

    int64_t getFwdConvPaddedImageDim(int64_t tensorDim, int64_t pad) {
        return tensorDim + (2 * pad);
    }
}

int64_t Utils::getFwdConvOutputDim(
    int64_t tensorDim,
    int64_t pad,
    int64_t filterDim,
    int64_t stride,
    int64_t dilation)
{
    int64_t p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
    return (p);
}

void Utils::generateStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat) {
    // For INT8x4 and INT8x32 we still compute standard strides here to input
    // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
    if (filterFormat == CUDNN_TENSOR_NCHW) {
        strideA[nbDims - 1] = 1;
        for (int64_t d = nbDims - 2; d >= 0; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
    }
    else {
        // Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1] = 1;
        strideA[nbDims - 1] = strideA[1] * dimA[1];
        for (int64_t d = nbDims - 2; d >= 2; d--) {
            strideA[d] = strideA[d + 1] * dimA[d + 1];
        }
        strideA[0] = strideA[2] * dimA[2];
    }
}

void Utils::checkCudaError(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        std::cout << std::format("CUDA ERROR OCCURED") << std::endl;
        assert(false);
    }
}

void Utils::checkCudnnError(cudnnStatus_t status)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        std::cout << std::format("CUDA ERROR OCCURED: {}", cudnnGetErrorString(status)) << std::endl;
        assert(false);
    }
}

namespace Utils
{
    cudnn_frontend::Tensor flattenTensor(cudnn_frontend::Tensor& tensor, int64_t id)
    {
        // TODO: magic numbers remove I shall
        if (tensor.getDimCount() > 3)
        {
            auto tensorDim = tensor.getDim();
            int64_t flattenTensorDim[] = { tensorDim[0], 1, tensorDim[1] * tensorDim[2] * tensorDim[3] }; // can be done better

            // TODO: Defaults, place
            constexpr int64_t alignment = 16; //16
            //constexpr cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
            constexpr cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
            constexpr float alpha = 1.0f;
            constexpr float beta = 0.0f;
            const int64_t FCstride[3] = { flattenTensorDim[1] * flattenTensorDim[2], flattenTensorDim[2], 1 };
            //Helpers::generateStrides(flattenTensorDim, FCstride, 3, tensorFormat);
            // 
            // RVO
            return cudnn_frontend::TensorBuilder()
                .setDim(3, flattenTensorDim)
                .setStride(3, FCstride)
                .setId(id)
                .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();
        }
        else
        {
            // RVO
            return cudnn_frontend::TensorBuilder()
                .setDim(3, tensor.getDim())
                .setStride(3, tensor.getStride())
                .setId(id)
                .setAlignment(tensor.getAlignment())
                .setDataType(static_cast<cudnnDataType_t>(tensor.getDataType()))
                .build();
        }
    }
}