#include "Pool.h"
//module NeuralNetwork:Pool;

import <format>;
import <iostream>;


Pool::Pool(cudnnHandle_t& handle, 
    Layer* previousLayer, 
    const Hyperparameters& hyperparameters,
    std::string name,
    VERBOSITY verbosity,
    const int64_t windowSize,
    const int64_t stride,
    const int64_t pad)
    : Layer(handle, previousLayer, hyperparameters, std::move(name), verbosity)
    , mWindowDim(mNbSpatialDims, windowSize)
    , mPrePadding(mNbSpatialDims, pad)
    , mPostPadding(mNbSpatialDims, pad)
    , mStride(mNbSpatialDims, stride)
{
    int64_t poolTensorDim[] = { 0, 0, 0, 0 };
    const auto& inputTensor = mPreviousLayer->getOutputTensor();
    const auto inputDim = inputTensor.getDim();
    poolTensorDim[0] = inputDim[0];
    poolTensorDim[1] = inputDim[1];
    poolTensorDim[2] = inputDim[2] / 2;
    poolTensorDim[3] = inputDim[3] / 2;

    if (mVerbosityLevel >= VERBOSITY::INFO)
    {
        std::cout << std::format("Creating Pool Layer {}", mName) << std::endl;
    }

    if (mVerbosityLevel >= VERBOSITY::REACH_INFO)
    {
        std::cout << std::format("After pool dims are {}, {}, {}, {}", poolTensorDim[0], poolTensorDim[1], poolTensorDim[2], poolTensorDim[3]) << std::endl;
    }

    const int64_t Psize = poolTensorDim[0] * poolTensorDim[1] * poolTensorDim[2] * poolTensorDim[3];
    mOutputSurface = std::make_unique<Surface<float>>(Psize, 0.0f);
    mGradSurface = std::make_unique<Surface<float>>(Psize, 0.0f);

    if (mVerbosityLevel >= VERBOSITY::REACH_INFO)
    {
        std::cout << inputTensor.describe() << std::endl;
    }

    try
    {
        constexpr int64_t nbDims = 4;

        mOutputTensor = std::make_unique<cudnn_frontend::Tensor>(Utils::createTensor(nbDims, poolTensorDim, generateTensorId()));
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << mOutputTensor->describe() << std::endl;

        // Define the resample descriptor
        auto poolDesc = cudnn_frontend::ResampleDescBuilder()
            .setComputeType(mDataType)
            .setNanPropagation(mNanOpt)
            .setResampleMode(mPoolMode)
            .setPaddingMode(mPaddingMode)
            .setSpatialDim(mNbSpatialDims, mWindowDim.data())
            .setSpatialStride(mNbSpatialDims, mStride.data())
            .setPrePadding(mNbSpatialDims, mPrePadding.data())
            .setPostPadding(mNbSpatialDims, mPostPadding.data())
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "Initialized Pool Desc" << std::endl;
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << poolDesc.describe() << std::endl;

        // Create a Resample Node
        auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
            .setxDesc(inputTensor)
            .setyDesc(*mOutputTensor)
            .setResampleDesc(poolDesc)
            .setAlpha(mAlpha)
            .setBeta(mBeta)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << pool_op.describe() << std::endl;

        std::vector<cudnn_frontend::Operation const*> ops = { &pool_op };
        std::vector<void*> data_ptrs;
        data_ptrs.emplace_back(mPreviousLayer->getOutputSurface().devPtr);
        data_ptrs.emplace_back(mOutputSurface->devPtr);

        std::vector<int64_t> uids;
        uids.emplace_back(inputTensor.getId());
        uids.emplace_back(mOutputTensor->getId());

        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "Setting forward propagation" << std::endl;
        _setForwardPropagationPlan(ops, data_ptrs, uids);
    }
    catch (cudnn_frontend::cudnnException& e) {
        if (mVerbosityLevel >= VERBOSITY::ERROR) std::cout << "[ERROR] Exception " << e.what() << std::endl;
        assert(false);
    }

    _setupGrad();
}

void Pool::propagateBackward()
{
    if (!mDataGradPlan || !mDataGradVariantPack)
    {
        return; // not initialized
    }

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        std::cout << std::format("Executing propagateForward on {}", mName.c_str()) << std::endl;
    }

    Utils::checkCudnnError(cudnnBackendExecute(mHandle, mDataGradPlan->get_raw_desc(), mDataGradVariantPack->get_raw_desc()));

}

void Pool::_setupGrad()
{
    auto& inputTensor = mPreviousLayer->getOutputTensor();
    constexpr int64_t nbDims = 4;

    try
    {

        auto inputGradTensor = Utils::createTensor(nbDims, inputTensor.getDim(), generateTensorId());
        auto gradTensor = Utils::createTensor(nbDims, mOutputTensor->getDim(), generateTensorId());

        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "Setting poolBwdDesc" << std::endl;
        auto poolBwdDesc = cudnn_frontend::ResampleDescBuilder()
            .setComputeType(mDataType)
            .setNanPropagation(mNanOpt)
            .setResampleMode(mPoolMode)
            .setPaddingMode(mPaddingMode)
            .setSpatialDim(mNbSpatialDims, mWindowDim.data())
            .setSpatialStride(mNbSpatialDims, mStride.data())
            .setPrePadding(mNbSpatialDims, mPrePadding.data())
            .setPostPadding(mNbSpatialDims, mPostPadding.data())
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << poolBwdDesc.describe() << std::endl;
        // Create a Resample Node
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "Setting pool_bwd_op" << std::endl;
        auto pool_bwd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
            .setdyDesc(gradTensor)
            .setdxDesc(inputGradTensor)
            .setResampleDesc(poolBwdDesc)
            .setAlpha(mAlpha)
            .setBeta(mBeta)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << pool_bwd_op.describe() << std::endl;

        std::vector<cudnn_frontend::Operation const*> ops_bwd = { &pool_bwd_op };
        std::vector<void*> data_ptrs_bwd;
        data_ptrs_bwd.emplace_back(mPreviousLayer->getGradSurface().devPtr);
        data_ptrs_bwd.emplace_back(mGradSurface->devPtr);

        std::vector<int64_t> uids_bwd;
        uids_bwd.emplace_back(inputGradTensor.getId());
        uids_bwd.emplace_back(gradTensor.getId());

        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "Setting back propagation" << std::endl;
        _setPlan(ops_bwd, data_ptrs_bwd, uids_bwd, mDataGradPlan, mDataGradVariantPack, mDataGradWorkspaceSize, mDataGradWorkspacePtr);
    }
    catch (cudnn_frontend::cudnnException& e) {
        if (mVerbosityLevel >= VERBOSITY::ERROR) std::cout << "[ERROR] Exception " << e.what() << std::endl;
        assert(false);
    }
}
