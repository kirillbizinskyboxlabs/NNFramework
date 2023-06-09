#include "Softmax.h"
//module;

#include "DevUtils.h"
#include <iomanip>

//module NeuralNetwork:Softmax;

import <format>;
import <iostream>;
//#include <iomanip>

Softmax::Softmax(cudnnHandle_t& handle, 
    Layer* previousLayer, 
    const Hyperparameters& hyperparameters,
    std::string name,
    VERBOSITY verbosity)
    : Layer(handle, previousLayer, hyperparameters, std::move(name), verbosity)
{
    // Can be moved to Layer??
    if (mVerbosityLevel >= VERBOSITY::INFO)
    {
        std::cout << std::format("Creating {} Layer", mName) << std::endl;
    }

    // We expect to have a flatten input // Add sanity check?

    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&sftTensorDesc);

    // TODO: Proper defaults
    constexpr int nbDims = 3;

    //auto& inputTensor = mPreviousLayer->getOutputTensor();
    auto inputTensor = Utils::flattenTensor(mPreviousLayer->getOutputTensor(), generateTensorId());

    //assert(nbDims == inputTensor.getDimCount());

    auto inputDim = inputTensor.getDim();
    auto inputStride = inputTensor.getStride();

    mDims = { static_cast<int>(inputDim[0]), static_cast<int>(inputDim[1]), static_cast<int>(inputDim[2]) };
    int stride[] = { mDims[1] * mDims[2] , mDims[2], 1 };
   
    int64_t size = std::accumulate(mDims.begin(), mDims.end(), 1ll, std::multiplies<int64_t>());

    if (mVerbosityLevel >= VERBOSITY::DEBUG) std::cout << std::endl << size << std::endl << inputTensor.describe() << std::endl;

    if (mVerbosityLevel == VERBOSITY::REACH_INFO)
    {
        std::cout << mPreviousLayer->getOutputTensor().describe() << std::endl;
    }

    mOutputSurface = std::make_unique<Surface<float>>(size, 0.0f);

    // TODO: rethink. They are identical and can propably be used interchangeably
    cudnnSetTensorNdDescriptor(srcTensorDesc,
        CUDNN_DATA_FLOAT,
        nbDims,
        mDims.data(),
        stride);
    cudnnSetTensorNdDescriptor(sftTensorDesc,
        CUDNN_DATA_FLOAT,
        nbDims,
        mDims.data(),
        stride);

    mOutputTensor = std::make_unique<cudnn_frontend::Tensor>(Utils::createTensor(nbDims, inputDim, generateTensorId()));


    if (mVerbosityLevel == VERBOSITY::REACH_INFO)
    {
        std::cout << mOutputTensor->describe() << std::endl;
    }

    // gradient surface has same dimensions as the output
    mGradSurface = std::make_unique<Surface<float>>(size, 0.0f);
}

void Softmax::propagateForward()
{
    if (mVerbosityLevel >= VERBOSITY::DEBUG) std::cout << "Softmax propagate Forward" << std::endl;
    // TODO: Proper defaults
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    auto status = cudnnSoftmaxForward(mHandle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        srcTensorDesc,
        mPreviousLayer->getOutputSurface().devPtr,
        &beta,
        sftTensorDesc,
        mOutputSurface->devPtr);

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        std::cout << cudnnGetErrorString(status) << std::endl;
        printOutput();
    }
}

void Softmax::printOutput()
{
    //TODO: deduplicate

    if (!mOutputSurface)
    {
        throw;
    }

    constexpr int precision = 8;

    mPreviousLayer->getOutputSurface().devToHostSync();
    mOutputSurface->devToHostSync();

    for (size_t i = 0; i < mDims.size(); ++i)
    {
        std::cout << std::format("yDim[{}]: {}", i, mDims[i]) << std::endl;
    }

    for (size_t i = 0; i < mDims[0]; ++i)
    {
        size_t batchStride = mDims[1] * mDims[2];

        std::cout << std::format("X{}", i) << std::endl;
        for (size_t j = 0; j < batchStride; ++j)
        {
            std::cout << std::setprecision(precision) << mPreviousLayer->getOutputSurface().hostPtr[batchStride * i + j] << " ";
        }

        std::cout << std::format("\nY{}", i) << std::endl;
        for (size_t j = 0; j < batchStride; ++j)
        {
            std::cout << std::setprecision(precision) << mOutputSurface->hostPtr[batchStride * i + j] << " ";
        }
        std::cout << std::endl;

    }
}

void Softmax::propagateBackward()
{
    // TODO: Proper defaults
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        std::cout << "Softmax backprop" << std::endl;
        printGrad();
    }

    Utils::checkCudnnError(cudnnSoftmaxBackward(
        mHandle,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        sftTensorDesc,
        mOutputSurface->devPtr,
        /*dyDesc*/ sftTensorDesc, // srcTensorDesc == sftTensorDesc, probably can be used interchangeably
        /**dy*/ mGradSurface->devPtr, // we expect valid gradient to be populated. make a sanity check?
        &beta,
        srcTensorDesc, // that makes sense, right? it describes the data pointer in the other layer
        mPreviousLayer->getGradSurface().devPtr)); // next level in backprop chain needs the gradient

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        printGrad();
    }
}