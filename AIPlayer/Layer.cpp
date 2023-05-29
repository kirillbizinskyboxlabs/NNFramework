#include "Layer.h"

#include <cublas.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include "DevUtils.h"

import <iostream>;
import <format>;
import <exception>;
//import <iomanip>;

#include <iomanip>

constexpr float epsilon = 0.01f; // learning rate
constexpr int64_t alignment = 16; //16B to make Tensor cores work
constexpr cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
constexpr cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
constexpr int convDim = 2;
constexpr float alpha = 1.0f;
constexpr float beta = 0.0f;


using namespace Utils;

void display_flat(float* image, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << std::setprecision(8) << image[i] << " ";
    }

    std::cout << "\n";
}

Layer::Layer(cudnnHandle_t& handle, Layer* previousLayer, bool verbose, std::string name)
    : mHandle(handle)
    , mVerbose(verbose)
    , mForwardPropagationWorkspaceSize(0)
    , mForwardPropagationWorkspacePtr(nullptr)
    , mPreviousLayer(previousLayer)
    , mName(std::move(name))
{
}

Layer::~Layer()
{
    if (mForwardPropagationWorkspacePtr)
    {
        checkCudaError(cudaFree(mForwardPropagationWorkspacePtr));
        mForwardPropagationWorkspacePtr = nullptr;
    }
}

void Layer::propagateForward()
{
    if (!mForwardPropagationPlan || !mForwardPropagationVariantPack)
    {
        return; // not initialized
    }

    if (mVerbose)
    {
        std::cout << std::format("Executing propagateForward on {}", mName.c_str()) << std::endl;
    }

    checkCudnnError(cudnnBackendExecute(mHandle, mForwardPropagationPlan->get_raw_desc(), mForwardPropagationVariantPack->get_raw_desc()));

}

cudnn_frontend::Tensor& Layer::getOutputTensor() const
{
    assert(mOutputTensor); // Assert it's initialized
    return *mOutputTensor;
}

Layer::Surface<float>& Layer::getOutputSurface() const
{
    assert(mOutputSurface);
    return *mOutputSurface;
}

Layer::Surface<float>& Layer::getGradSurface() const
{
    assert(mGradSurface);
    return *mGradSurface;
}

void Layer::printOutput()
{
    if (!mVerbose)
    {
        return;
    }

    if (!mOutputSurface)
    {
        throw;
    }

    mOutputSurface->devToHostSync();

    /*cudaDeviceSynchronize();
    checkCudaError(cudaMemcpy(mOutputSurface->hostPtr, mOutputSurface->devPtr, (size_t)(sizeof(mOutputSurface->hostPtr[0]) * mOutputSurface->n_elems), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();*/

    auto yDim = mOutputTensor->getDim();
    for (size_t i = 0; i < CUDNN_DIM_MAX + 1; ++i)
    {
        std::cout << std::format("yDim[{}]: {}", i, yDim[i]) << std::endl;
    }

    if (yDim[2] == yDim[3])
    {
        cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
        int64_t stride[4];
        generateStrides(yDim, stride, 4, tensorFormat);
        for (int64_t h = 0; h < yDim[2]; ++h)
        {
            for (int64_t w = 0; w < yDim[3]; ++w)
            {
                std::cout << std::setw(3) << std::setprecision(3) << mOutputSurface->hostPtr[h * stride[2] + w * stride[3]] << " ";
            }
            std::cout << std::endl;
        }
    }
    else
    {
        display_flat(mOutputSurface->hostPtr, yDim[1]);
    }
}

void Layer::printGrad()
{
    if (!mVerbose)
    {
        return;
    }

    if (!mGradSurface || !mOutputTensor)
    {
        return;
    }

    mGradSurface->devToHostSync();

    auto nbDims = mOutputTensor->getDimCount();
    auto dims = mOutputTensor->getDim();
    auto stride = mOutputTensor->getStride();
    std::cout << std::format("{} dY:", mName.c_str()) << std::endl;

    if (dims[nbDims - 2] == 1 && dims[nbDims - 1] == 1)
    {
        for (int64_t b = 0; b < dims[0]; ++b)
        {
            for (int64_t f = 0; f < dims[1]; ++f)
            {
                std::cout << mGradSurface->hostPtr[stride[0] * b + stride[1] * f] << " ";
            }
            std::cout << std::endl;
        }
    }
    else
    {
        for (int64_t b = 0; b < dims[0]; ++b)
        {
            for (int64_t h = 0; h < dims[nbDims - 2]; ++h)
            {
                for (int64_t w = 0; w < dims[nbDims - 1]; ++w)
                {
                    std::cout << mGradSurface->hostPtr[stride[0] * b + stride[nbDims - 2] * h + stride[nbDims - 1] * w] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

void Layer::_setPlan(std::vector<cudnn_frontend::Operation const*>& ops,
    std::vector<void*> data_ptrs,
    std::vector<int64_t> uids,
    std::unique_ptr<cudnn_frontend::ExecutionPlan>& plan,
    std::unique_ptr<cudnn_frontend::VariantPack>& variantPack,
    int64_t& workspace_size,
    void*& workspace_ptr)
{
    if (mVerbose) std::cout << "_setPlan" << std::endl;

    auto opGraph = cudnn_frontend::OperationGraphBuilder()
        .setHandle(mHandle)
        .setOperationGraph(ops.size(), ops.data())
        .build();

    if (mVerbose) std::cout << opGraph.describe() << std::endl;

    plan = std::make_unique<cudnn_frontend::ExecutionPlan>(get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));

    if (mVerbose) std::cout << "Plan tag: " << plan->getTag() << std::endl;

    workspace_size = plan->getWorkspaceSize();
    if (mVerbose) std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

    if (workspace_size > 0) {
        checkCudaError(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
    }

    assert(data_ptrs.size() == uids.size());
    int64_t num_ptrs = data_ptrs.size();
    if (mVerbose) std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
    variantPack = std::make_unique<cudnn_frontend::VariantPack>(cudnn_frontend::VariantPackBuilder()
        .setWorkspacePointer(workspace_ptr)
        .setDataPointers(num_ptrs, data_ptrs.data())
        .setUids(num_ptrs, uids.data())
        .build());
    if (mVerbose) std::cout << "variantPack " << variantPack->describe() << std::endl;
}

void Layer::_setForwardPropagationPlan(std::vector<cudnn_frontend::Operation const*>& ops, std::vector<void*> data_ptrs, std::vector<int64_t> uids)
{
    _setPlan(ops, data_ptrs, uids, mForwardPropagationPlan, mForwardPropagationVariantPack, mForwardPropagationWorkspaceSize, mForwardPropagationWorkspacePtr);

}
