#include "Softmax.h"

#include "DevUtils.h"

import <format>;
import <iostream>;
#include <iomanip>

Softmax::Softmax(cudnnHandle_t& handle, 
    Layer* previousLayer, 
    bool verbose, 
    std::string name)
    : Layer(handle, previousLayer, verbose, std::move(name))
    //, mSrcSurface(prevLayer.getOutputSurface())
    //, mPrevLayer(prevLayer)
{
    // We expect to have a flatten input
    // Since we use backend here - we won't set a frontend descriptor
    //cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;

    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&sftTensorDesc);

    // TODO: Proper defaults
    constexpr int nbDims = 3;

    //auto& inputTensor = mPreviousLayer->getOutputTensor();
    auto inputTensor = Utils::flattenTensor(mPreviousLayer->getOutputTensor(), generateTensorId());

    //assert(nbDims == inputTensor.getDimCount());

    auto inputDim = inputTensor.getDim();
    //int nbDims = inputTensor.getDimCount();
    auto inputStride = inputTensor.getStride();
    //std::vector<int> stride;
    //// We don't expect to have more than INT_MAX elements in dimensions, so this should be safe... Rigth?
    //for (int64_t d = 0; d < nbDims; ++d)
    //{
    //    mDims.emplace_back(static_cast<int>(inputDim[d]));
    //    stride.emplace_back(static_cast<int>(inputStride[d]));

    //    std::cout << std::format("{} {}, ", mDims.back(), stride.back());
    //}

    mDims = { static_cast<int>(inputDim[0]), static_cast<int>(inputDim[1]), static_cast<int>(inputDim[2]) };
    int stride[] = { mDims[1] * mDims[2] , mDims[2], 1 };


    //mDims = { static_cast<int>(inputDim[0]), static_cast<int>(inputDim[1]), static_cast<int>(inputDim[2]) };
    //int stride[] = { mDims[1] * mDims[2] , mDims[2], 1 };
    
    int64_t size = std::accumulate(mDims.begin(), mDims.end(), 1ll, std::multiplies<int64_t>());

    if (mVerbose) std::cout << std::endl << size << std::endl << inputTensor.describe() << std::endl;

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

    // we need to define output
    mOutputTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
        .setDim(nbDims, inputDim)
        .setStride(nbDims, inputTensor.getStride())
        .setId(generateTensorId())
        .setAlignment(16)
        .setDataType(CUDNN_DATA_FLOAT)
        .build());

    // gradient surface has same dimensions as the output
    mGradSurface = std::make_unique<Surface<float>>(size, 0.0f);
}

void Softmax::propagateForward()
{
    if (mVerbose) std::cout << "Softmax propagate Forward" << std::endl;
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

    if (mVerbose)
    {
        std::cout << cudnnGetErrorString(status) << std::endl;
    }
}

void Softmax::printOutput()
{
    //TODO: deduplicate

    if (!mOutputSurface)
    {
        throw;
    }

    mPreviousLayer->getOutputSurface().devToHostSync();
    mOutputSurface->devToHostSync();

    //auto yDim = yTensor->getDim();
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
            std::cout << std::setprecision(3) << mPreviousLayer->getOutputSurface().hostPtr[batchStride * i + j] << " ";
        }

        std::cout << std::format("\nY{}", i) << std::endl;
        for (size_t j = 0; j < batchStride; ++j)
        {
            std::cout << std::setprecision(3) << mOutputSurface->hostPtr[batchStride * i + j] << " ";
        }
        std::cout << std::endl;

    }
}

void Softmax::propagateBackward()
{
    // TODO: Proper defaults
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    //printGrad();
    if (mVerbose)
    {
        std::cout << "Softmax backprop" << std::endl;
    }

    try {
        auto status = cudnnSoftmaxBackward(
            mHandle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            sftTensorDesc,
            mOutputSurface->devPtr,
            /*dyDesc*/ srcTensorDesc, // srcTensorDesc == sftTensorDesc, probably can be used interchangeably
            /**dy*/ mGradSurface->devPtr, // we expect valid gradient to be populated. make a sanity check?
            &beta,
            srcTensorDesc, // that makes sense, right? it describes the data pointer in the other layer
            mPreviousLayer->getGradSurface().devPtr); // next level in backprop chain needs the gradient

        if (mVerbose)
        {
            std::cout << cudnnGetErrorString(status) << std::endl;
        }
    }
    catch (cudnn_frontend::cudnnException& e)
    {
        std::cout << std::format("Softmax backprop failed: {}", e.what()) << std::endl;
    }
}