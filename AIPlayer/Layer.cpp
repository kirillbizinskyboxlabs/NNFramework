//module;
#include "Layer.h"

#include <cublas.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include "DevUtils.h"
#include <iomanip>

//module NeuralNetwork:Layer;

import <iostream>;
import <format>;
import <exception>;
//import <iomanip>; // didn't work

constexpr int precision = 8;

using namespace Utils;

void display_flat(float* image, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << std::setprecision(8) << image[i] << " ";
    }

    std::cout << "\n";
}

Layer::Layer(cudnnHandle_t& handle, Layer* previousLayer, const Hyperparameters& hyperparameters, std::string name, VERBOSITY verbosityLevel)
    : mHandle(handle)
    , mForwardPropagationWorkspaceSize(0)
    , mForwardPropagationWorkspacePtr(nullptr)
    , mPreviousLayer(previousLayer)
    , mName(std::move(name))
    , mVerbosityLevel(verbosityLevel)
    , mHyperparameters(hyperparameters)
{
}

Layer::~Layer()
{
    if (mForwardPropagationWorkspacePtr)
    {
        checkCudaError(cudaFree(mForwardPropagationWorkspacePtr));
        mForwardPropagationWorkspacePtr = nullptr;
    }

    if (mDataGradWorkspacePtr)
    {
        checkCudaError(cudaFree(mDataGradWorkspacePtr));
        mDataGradWorkspacePtr = nullptr;
    }
}

void Layer::propagateForward()
{
    if (!mForwardPropagationPlan || !mForwardPropagationVariantPack)
    {
        return; // not initialized
    }

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        std::cout << std::format("Executing propagateForward on {}", mName.c_str()) << std::endl;
    }

    checkCudnnError(cudnnBackendExecute(mHandle, mForwardPropagationPlan->get_raw_desc(), mForwardPropagationVariantPack->get_raw_desc()));

}

const cudnn_frontend::Tensor& Layer::getOutputTensor() const
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
    if (mVerbosityLevel < VERBOSITY::DEBUG)
    {
        return;
    }

    if (!mOutputSurface)
    {
        throw;
    }

    mOutputSurface->devToHostSync();

    auto yDim = mOutputTensor->getDim();
    for (size_t i = 0; i < CUDNN_DIM_MAX + 1; ++i)
    {
        std::cout << std::format("yDim[{}]: {}", i, yDim[i]) << std::endl;
    }

    auto stride = mOutputTensor->getStride();

    if (yDim[2] == yDim[3] && yDim[2] != 1)
    {
        for (int64_t b = 0; b < yDim[0]; ++b)
        {
            for (int64_t c = 0; c < yDim[1]; ++c)
            {
                for (int64_t h = 0; h < yDim[2]; ++h)
                {
                    for (int64_t w = 0; w < yDim[3]; ++w)
                    {
                        std::cout << std::setw(3) << std::setprecision(precision) << mOutputSurface->hostPtr[b*stride[0] + c*stride[1] + h * stride[2] + w * stride[3]] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
        
    }
    else
    {
        for (int64_t b = 0; b < yDim[0]; ++b)
        {
            for (int64_t c = 0; c < yDim[1]; ++c)
            {
                std::cout << std::setw(3) << std::setprecision(precision) << mOutputSurface->hostPtr[b * stride[0] + c] << " ";
            }
            std::cout << std::endl;
        }
    }
}

void Layer::printGrad()
{
    if (mVerbosityLevel < VERBOSITY::DEBUG)
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
                    std::cout << std::setprecision(precision) << mGradSurface->hostPtr[stride[0] * b + stride[nbDims - 2] * h + stride[nbDims - 1] * w] << " ";
                }
                std::cout << std::endl;
            }
        }
    }
}

void Layer::_setPlan(std::vector<cudnn_frontend::Operation const*>& ops,
                     std::vector<void*>& data_ptrs,
                     std::vector<int64_t>& uids,
                     std::unique_ptr<cudnn_frontend::ExecutionPlan>& plan,
                     std::unique_ptr<cudnn_frontend::VariantPack>& variantPack,
                     int64_t& workspace_size,
                     void*& workspace_ptr)
{
    if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "_setPlan" << std::endl;

    auto opGraph = cudnn_frontend::OperationGraphBuilder()
        .setHandle(mHandle)
        .setOperationGraph(ops.size(), ops.data())
        .build();

    if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << opGraph.describe() << std::endl;

    plan = std::make_unique<cudnn_frontend::ExecutionPlan>(get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));

    if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "Plan tag: " << plan->getTag() << std::endl;

    workspace_size = plan->getWorkspaceSize();
    if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

    if (workspace_size > 0) {
        checkCudaError(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
    }

    assert(data_ptrs.size() == uids.size());
    int64_t num_ptrs = data_ptrs.size();
    if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
    variantPack = std::make_unique<cudnn_frontend::VariantPack>(cudnn_frontend::VariantPackBuilder()
        .setWorkspacePointer(workspace_ptr)
        .setDataPointers(num_ptrs, data_ptrs.data())
        .setUids(num_ptrs, uids.data())
        .build());
    if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "variantPack " << variantPack->describe() << std::endl;
}

void Layer::_setForwardPropagationPlan(std::vector<cudnn_frontend::Operation const*>& ops, std::vector<void*>& data_ptrs, std::vector<int64_t>& uids)
{
    _setPlan(ops, data_ptrs, uids, mForwardPropagationPlan, mForwardPropagationVariantPack, mForwardPropagationWorkspaceSize, mForwardPropagationWorkspacePtr);
}
