#pragma once

//enum cudnnStatus_t;
//enum cudaError_t;
#include <cublas.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
//#include <cudnn_frontend_ExecutionPlan.h>

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

    cudnn_frontend::Tensor flattenTensor(cudnn_frontend::Tensor& tensor, int64_t id);
}