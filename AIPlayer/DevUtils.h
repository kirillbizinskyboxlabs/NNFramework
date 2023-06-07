#pragma once

#include <cublas.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

enum class VERBOSITY
{
    NONE = 0,

    MIN,
    ERROR,
    INFO,
    REACH_INFO,
    DEBUG,

    MAX
};

namespace Utils
{
	void initImage(float* image, int64_t imageSize);
	cudnn_frontend::ExecutionPlan get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_);

    int64_t getFwdConvOutputDim(
        int64_t tensorDim,
        int64_t pad,
        int64_t filterDim,
        int64_t stride,
        int64_t dilation);

    void generateStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat);
    void checkCudaError(cudaError_t status);
    void checkCudnnError(cudnnStatus_t status);

    cudnn_frontend::Tensor createTensor(
        int64_t nbDims,
        const int64_t* dims, // send vector instead?
        int64_t id,
        bool isVirtual = false,
        cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC,
        cudnnDataType_t dataType = CUDNN_DATA_FLOAT,
        int64_t alignment = 16);

    cudnn_frontend::Tensor flattenTensor(const cudnn_frontend::Tensor& tensor, int64_t id);
}

struct Hyperparameters
{
    enum class UpdateType
    {
        SGD,
        mSGD,
        ADAM
    };

    UpdateType updateType = UpdateType::SGD;

    //cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    //cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    //int64_t alignment = 16;

    //int64_t nbDims = 4; // questionable

    // name duplication. Not good
    float alpha = 1.0f;
    float beta = 0.0f;

    float biasStartValue = 0.0001f;

    // defaults
    const int64_t dilationA[2] = { 1, 1 };
    const int64_t convstrideA[2] = { 1, 1 }; // fun: array bound cannot be deduced from a default member initializer

    struct
    {
        float alpha = 0.001f; // value from ADAM paper
        //float alpha = 0.00001; // value from DIM paper
        float alpha_t; // this will be calcualted at each step
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float epsilon = 1.0e-8f;
        float epsilon_t;
        size_t t = 0;
    } adam;

    struct
    {
        float momentum = 0.9f; // momentum
        float L2 = 0.0005f; // L2 or weight decay
        float lr = 0.001f; // learning rate
    } msgd;

};