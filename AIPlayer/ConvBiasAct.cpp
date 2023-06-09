#include "ConvBiasAct.h"

import <format>;
import <iostream>;
import <random>;
#include <iomanip>


ConvBiasAct::ConvBiasAct(cudnnHandle_t& handle,
    const int64_t kernelSize,
    const int64_t filterSize,
    Layer* previousLayer,
    const Hyperparameters& hyperparameters,
    const float& learningRate,
    const int64_t convPad,
    bool training,
    bool needDataGrad,
    std::string name,
    VERBOSITY verbosity,
    const int64_t dilation,
    const int64_t convStride)
    : Layer(handle, previousLayer, hyperparameters, std::move(name), verbosity)
    //, mPadHeight(convPad)
    //, mPadWidth(convPad)
    , mPad(mConvDim, convPad)
    , mDilation(mConvDim, dilation)
    , mConvStride(mConvDim, convStride)
    , mFilterSize(filterSize)
    , mKernelSize(kernelSize)
    , mLearningRate(learningRate)
    , mNeedDataGrad(needDataGrad)
{
    // Defaults. TODO: Move to the appropriate place. Some of these are hyperparameters. Some are necessary constants. Bad place.
    //constexpr int64_t alignment = 16; //16B to make Tensor cores work
    //const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    //const cudnnConvolutionMode_t convMode = CUDNN_CROSS_CORRELATION;
    //constexpr int convDim = 2;
    //constexpr float alpha = 1.0f;
    //constexpr float beta = 0.0f;

    constexpr int64_t nbDims = 4;

    auto inputDim = mPreviousLayer->getOutputTensor().getDim();

    const int64_t wTensorDim[] = { filterSize, inputDim[1], kernelSize, kernelSize }; // filter
    int64_t yTensorDim[] = { 0, 0, 0, 0 }; // Computed Below
    //const int64_t padA[] = { convPad, convPad };
    //const int64_t dilationA[] = { 1, 1 }; // TODO: make proper defaults
    //const int64_t convstrideA[] = { 1, 1 };
    const int64_t bTensorDim[] = { 1, wTensorDim[0], 1, 1 };  // bias

    yTensorDim[0] = inputDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        //yTensorDim[dim + 2] = Utils::getFwdConvOutputDim(inputDim[dim + 2], padA[dim], wTensorDim[dim + 2], convstrideA[dim], dilationA[dim]);
        yTensorDim[dim + 2] = Utils::getFwdConvOutputDim(inputDim[dim + 2], mPad[dim], wTensorDim[dim + 2], mConvStride[dim], mDilation[dim]);
    }

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    // Heperparameters
    //constexpr float biasStartValue = 0.0001f;
    //constexpr float XavierNumerator = 3.0f;

    // Xavier filter values
    auto XavierInitializer = [inputChannels = inputDim[1], kernelSize, XavierNumerator = mHyperparameters.XavierNumerator](size_t size, float* hostPtr)
    {
        std::random_device dev;
        std::mt19937 gen(dev());
        const float deviation = sqrt(XavierNumerator / (inputChannels * kernelSize * kernelSize));
        std::uniform_real_distribution<float> distribution(-deviation, deviation);

        for (size_t i = 0; i < size; ++i)
        {
            hostPtr[i] = distribution(gen);
        }
    };

    mWeightsSurface = std::make_unique<Surface<float>>(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], XavierInitializer);
    mBiasSurface = std::make_unique<Surface<float>>(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], mHyperparameters.biasStartValue);
    mOutputSurface = std::make_unique<Surface<float>>(Ysize, 0.0f);

    if (mVerbosityLevel >= VERBOSITY::INFO)
    {
        std::cout << std::format("Creating Convolution Layer {} with {} parameters", mName, mWeightsSurface->n_elems + mBiasSurface->n_elems) << std::endl;
    }

    if (mVerbosityLevel >= VERBOSITY::REACH_INFO)
    {
        std::cout << std::format("====DIMENSIONS====") << std::endl;
        std::cout << std::format("input dims are {}, {}, {}, {}", inputDim[0], inputDim[1], inputDim[2], inputDim[3]) << std::endl;
        std::cout << std::format("filter dims are {}, {}, {}, {}", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]) << std::endl;
        std::cout << std::format("output dims are {}, {}, {}, {}", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]) << std::endl;
    }

    //std::random_device dev;
    //std::mt19937 gen(dev());
    //const float deviation = sqrt(3.0f / (inputDim[1] * kernelSize * kernelSize)); // input channels * KS^2
    ////const float deviation = 1.0f / (inputDim[1] * kernelSize * kernelSize); // input channels * KS^2
    //std::uniform_real_distribution<float> distribution(-deviation, deviation);

    //for (int64_t i = 0; i < mWeightsSurface->n_elems; ++i)
    //{
    //    mWeightsSurface->hostPtr[i] = distribution(gen);
    //}

    //mWeightsSurface->hostToDevSync();

    try
    {
        //int64_t stride[nbDims];
        //Utils::generateStrides(wTensorDim, stride, nbDims, tensorFormat);
        //auto wTensor = cudnn_frontend::TensorBuilder()
        //    .setDim(nbDims, wTensorDim)
        //    .setStride(nbDims, stride)
        //    .setId(generateTensorId())
        //    .setAlignment(alignment)
        //    .setDataType(dataType)
        //    .build();

        auto filterTensor = Utils::createTensor(nbDims, wTensorDim, generateTensorId());

        //Utils::generateStrides(bTensorDim, stride, nbDims, tensorFormat);
        //auto bTensor = cudnn_frontend::TensorBuilder()
        //    .setDim(nbDims, bTensorDim)
        //    .setStride(nbDims, stride)
        //    .setId(generateTensorId())
        //    .setAlignment(alignment)
        //    .setDataType(dataType)
        //    .build();

        auto biasTensor = Utils::createTensor(nbDims, bTensorDim, generateTensorId());

        //Utils::generateStrides(yTensorDim, stride, nbDims, tensorFormat);
        //auto afterConvTensor = cudnn_frontend::TensorBuilder()
        //    .setDim(nbDims, yTensorDim)
        //    .setStride(nbDims, stride)
        //    .setId(generateTensorId())  // after conv
        //    .setAlignment(alignment)
        //    .setVirtual()
        //    .setDataType(dataType)
        //    .build();

        auto afterConvTensor = Utils::createTensor(nbDims, yTensorDim, generateTensorId(), true);

        //auto afterBiasTensor = cudnn_frontend::TensorBuilder()
        //    .setDim(nbDims, yTensorDim)
        //    .setStride(nbDims, stride)
        //    .setId(generateTensorId())  // after bias
        //    .setAlignment(alignment)
        //    .setVirtual()
        //    .setDataType(dataType)
        //    .build();

        auto afterBiasTensor = Utils::createTensor(nbDims, yTensorDim, generateTensorId(), true);

        //mOutputTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
        //    .setDim(nbDims, yTensorDim)
        //    .setStride(nbDims, stride)
        //    .setId(generateTensorId())  // after relu
        //    .setAlignment(alignment)
        //    .setDataType(dataType)
        //    .build());

        // after activation
        mOutputTensor = std::make_unique<cudnn_frontend::Tensor>(Utils::createTensor(nbDims, yTensorDim, generateTensorId()));

        if (mVerbosityLevel >= VERBOSITY::REACH_INFO)
        {
            std::cout << mPreviousLayer->getOutputTensor().describe() << std::endl;
            std::cout << filterTensor.describe() << std::endl;
            std::cout << biasTensor.describe() << std::endl;
            std::cout << afterConvTensor.describe() << std::endl;
            std::cout << afterBiasTensor.describe() << std::endl;
            std::cout << mOutputTensor->describe() << std::endl;
        }

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
            .setComputeType(mDataType)
            .setMathMode(mConvMode)
            .setSpatialDimCount(mConvDim)
            .setSpatialStride(mConvDim, mConvStride.data())
            .setPrePadding(mConvDim, mPad.data())
            .setPostPadding(mConvDim, mPad.data())
            .setDilation(mConvDim, mDilation.data())
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << convDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_ADD)
            .setComputeType(mDataType)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_RELU_FWD)// TODO: hardcoded -> parametrized. Can be CUDNN_POINTWISE_SIGMOID_FWD even though ReLU is proven to be superior
            .setComputeType(mDataType)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << actDesc.describe() << std::endl;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            .setxDesc(mPreviousLayer->getOutputTensor())
            //.setwDesc(wTensor)
            .setwDesc(filterTensor)
            .setyDesc(afterConvTensor) 
            .setcDesc(convDesc)
            .setAlpha(mAlpha)
            .setBeta(mBeta)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << conv_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(conv_op.getOutputTensor())
            //.setbDesc(bTensor)
            .setbDesc(biasTensor)
            .setyDesc(afterBiasTensor)
            .setpwDesc(biasDesc)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(bias_op.getOutputTensor())
            .setyDesc(*mOutputTensor)
            .setpwDesc(actDesc)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << act_op.describe() << std::endl;

        std::vector<cudnn_frontend::Operation const*> ops;
        ops.emplace_back(&conv_op);
        ops.emplace_back(&bias_op);
        ops.emplace_back(&act_op);

        std::vector<void*> data_ptrs;
        data_ptrs.emplace_back(mPreviousLayer->getOutputSurface().devPtr);
        data_ptrs.emplace_back(mWeightsSurface->devPtr);
        data_ptrs.emplace_back(mBiasSurface->devPtr);
        data_ptrs.emplace_back(mOutputSurface->devPtr);

        std::vector<int64_t> uids;
        uids.emplace_back(mPreviousLayer->getOutputTensor().getId());
        //uids.emplace_back(wTensor.getId());
        uids.emplace_back(filterTensor.getId());
        //uids.emplace_back(bTensor.getId());
        uids.emplace_back(biasTensor.getId());
        uids.emplace_back(mOutputTensor->getId());

        _setPlan(ops, data_ptrs, uids, mForwardPropagationPlan, mForwardPropagationVariantPack, mForwardPropagationWorkspaceSize, mForwardPropagationWorkspacePtr);
    }
    catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
    }

    if (training)
    {
        //mGradSurface = std::make_unique<Surface<float>>(Ysize, 0.0f);
        //Utils::checkCudnnError(cudnnCreateFilterDescriptor(&mFilterDesc));
        //Utils::checkCudnnError(cudnnSetFilter4dDescriptor(mFilterDesc,
        //    /*dataType=*/mDataType,
        //    /*format=*/mTensorFormat,
        //    /*out_channels=*/filterSize,
        //    /*in_channels=*/inputDim[1],
        //    /*kernel_height=*/kernelSize,
        //    /*kernel_width=*/kernelSize)); // most of convolution requires square kernels - change if needed
        //Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mBiasGradTensorDesc));
        //Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mBiasGradTensorDesc,
        //    mTensorFormat,
        //    mDataType,
        //    1, filterSize,
        //    1, 1));
        //Utils::checkCudnnError(cudnnCreateConvolutionDescriptor(&mConvDesc));
        //Utils::checkCudnnError(cudnnSetConvolution2dDescriptor(mConvDesc,
        //    /*pad_height=*/convPad,
        //    /*pad_width=*/convPad,
        //    /*vertical_stride=*/1,
        //    /*horizontal_stride=*/1,
        //    /*dilation_height=*/1,
        //    /*dilation_width=*/1,
        //    /*mode=*/mConvMode,
        //    /*computeType=*/mDataType));
        ////cudnnGetConvolutionBackwardDataAlgorithm_v7
        ////cudnnConvolutionBwdDataPreference_t bwdDPref;
        ////cudnnConvolutionBwdFilterPreference_t bwdFPref;
        //Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mInputTensorDesc));
        //Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mGradTensorDesc));
        //Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mInputTensorDesc,
        //    /*format=*/mTensorFormat,
        //    /*dataType=*/mDataType,
        //    /*batch_size=*/inputDim[0],
        //    /*channels=*/inputDim[1],
        //    /*image_height=*/inputDim[2],
        //    /*image_width=*/inputDim[3]));
        //Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mGradTensorDesc,
        //    /*format=*/mTensorFormat,
        //    /*dataType=*/mDataType,
        //    /*batch_size=*/inputDim[0],
        //    /*channels=*/yTensorDim[1],
        //    /*image_height=*/yTensorDim[2],
        //    /*image_width=*/yTensorDim[3]));
        //
        //const int requestedAlgoCount = 1;
        //int returnedAlgoCount;
        //Utils::checkCudnnError(cudnnFindConvolutionBackwardDataAlgorithm(
        //    /*cudnnHandle_t                          */mHandle,
        //    /*const cudnnFilterDescriptor_t          */mFilterDesc,
        //    /*const cudnnTensorDescriptor_t          */mGradTensorDesc,
        //    /*const cudnnConvolutionDescriptor_t     */mConvDesc,
        //    /*const cudnnTensorDescriptor_t          */mInputTensorDesc,
        //    /*const int                              */requestedAlgoCount,
        //    /*int* */&returnedAlgoCount,
        //    /*cudnnConvolutionBwdDataAlgoPerf_t* */&mBwdDPerf));
        //Utils::checkCudnnError(cudnnFindConvolutionBackwardFilterAlgorithm(
        //    mHandle, 
        //    mInputTensorDesc,
        //    mGradTensorDesc,
        //    mConvDesc,
        //    mFilterDesc,
        //    requestedAlgoCount,
        //    &returnedAlgoCount,
        //    &mBwdFPerf));
        //mDataGradWorkspaceSize = std::max(mBwdDPerf.memory, mBwdFPerf.memory);
        //if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << std::format("{} backpropagation descriptor setup completed. Workspace size: {}", mName, mDataGradWorkspaceSize) << std::endl;
        //if (mDataGradWorkspaceSize > 0)
        //{
        //    Utils::checkCudaError(cudaMalloc(&mDataGradWorkspacePtr, mDataGradWorkspaceSize));
        //}

        _setupBackPropagation();
    }
}

ConvBiasAct::~ConvBiasAct()
{
    Utils::checkCudnnError(cudnnDestroyTensorDescriptor(mInputTensorDesc));
    Utils::checkCudnnError(cudnnDestroyTensorDescriptor(mGradTensorDesc));
    Utils::checkCudnnError(cudnnDestroyTensorDescriptor(mBiasGradTensorDesc));
    Utils::checkCudnnError(cudnnDestroyFilterDescriptor(mFilterDesc));
    Utils::checkCudnnError(cudnnDestroyConvolutionDescriptor(mConvDesc));

    if (mBiasGradWorkspacePtr)
    {
        Utils::checkCudaError(cudaFree(mBiasGradWorkspacePtr));
        mForwardPropagationWorkspacePtr = nullptr;
    }

    if (mActivationGradWorkspacePtr)
    {
        Utils::checkCudaError(cudaFree(mActivationGradWorkspacePtr));
        mForwardPropagationWorkspacePtr = nullptr;
    }
}

void ConvBiasAct::propagateBackward()
{
    Utils::checkCudnnError(cudnnBackendExecute(mHandle, mActivationGradPlan->get_raw_desc(), mActivationGradVariantPack->get_raw_desc()));

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        std::cout << std::format("Propagating backwards on {}, learning rate: {}", mName, mLearningRate) << std::endl;

        _printBias();
        _printFilter();
        _printActivationGrad();
    }

     float alpha = 1.0f, beta = 0.0f; // not sure, maybe it can be a nn member

    Utils::checkCudnnError(cudnnConvolutionBackwardBias(
        mHandle, 
        &alpha, 
        mGradTensorDesc,
        //mGradSurface->devPtr, 
        mActivationGradSurface->devPtr,
        &beta, 
        mBiasGradTensorDesc,
        mBiasGradSurface->devPtr));

    Utils::checkCudnnError(cudnnConvolutionBackwardFilter(
        mHandle, 
        &alpha, 
        mInputTensorDesc,
        mPreviousLayer->getOutputSurface().devPtr,
        mGradTensorDesc,
        //mGradSurface->devPtr,
        mActivationGradSurface->devPtr,
        mConvDesc,
        mBwdFPerf.algo,
        mDataGradWorkspacePtr, 
        mDataGradWorkspaceSize,
        &beta, 
        mFilterDesc,
        mWeightsGradSurface->devPtr));

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        _printBiasGrad();
        _printFilterGrad();
    }

    if (mNeedDataGrad)
    {
        Utils::checkCudnnError(cudnnConvolutionBackwardData(mHandle, 
            &alpha, 
            mFilterDesc,
            mWeightsSurface->devPtr,
            mGradTensorDesc,
            //mGradSurface->devPtr,
            mActivationGradSurface->devPtr,
            mConvDesc,
            mBwdDPerf.algo,
            mDataGradWorkspacePtr, 
            mDataGradWorkspaceSize,
            &beta, 
            mInputTensorDesc,
            mPreviousLayer->getGradSurface().devPtr));
    }

    // Update weights
    //switch (mHyperparameters.updateType)
    if (mHyperparameters.updateType == Hyperparameters::UpdateType::SGD) // SGD
    {
        _SGDUpdate();
        //float alpha = -mLearningRate; // TODO: change name
        //cublasSaxpy(
        //    mWeightsSurface->n_elems,
        //    alpha, 
        //    mWeightsGradSurface->devPtr, 1,
        //    mWeightsSurface->devPtr, 1);
        //cublasSaxpy(
        //    mBiasSurface->n_elems,
        //    alpha, 
        //    mBiasGradSurface->devPtr, 1,
        //    mBiasSurface->devPtr, 1);
    }
    else if (mHyperparameters.updateType == Hyperparameters::UpdateType::mSGD)
    {
        _miniBatchSGDUpdate();
        //cublasSscal(
        //    mWeightsSurface->n_elems,
        //    mHyperparameters.msgd.momentum, 
        //    mSGD.mGradFilterVelocitySurface->devPtr, 
        //    1); // v = momentum * v
        //alpha = -mHyperparameters.msgd.L2 * mHyperparameters.msgd.lr; // alpha = -L2*epsilon
        //cublasSaxpy(
        //    mWeightsSurface->n_elems,
        //    alpha, 
        //    mWeightsSurface->devPtr, 1,
        //    mSGD.mGradFilterVelocitySurface->devPtr, 1); // v = -L2*epsilon*w + v
        //alpha = -mHyperparameters.msgd.lr; // alpha = -epsilon
        //cublasSaxpy(
        //    mWeightsSurface->n_elems,
        //    alpha, 
        //    mWeightsGradSurface->devPtr, 1,
        //    mSGD.mGradFilterVelocitySurface->devPtr, 1); // v = -epsilon*grad + v
        //alpha = 1;
        //cublasSaxpy(
        //    mWeightsSurface->n_elems,
        //    alpha, 
        //    mSGD.mGradFilterVelocitySurface->devPtr, 1,
        //    mWeightsSurface->devPtr, 1); // w = v + w

        //// bias
        //cublasSscal(
        //    mBiasSurface->n_elems,
        //    mHyperparameters.msgd.momentum, 
        //    mSGD.mGradBiasVelocitySurface->devPtr, 1); // v = momentum * v
        //alpha = -mHyperparameters.msgd.L2 * mHyperparameters.msgd.lr; // alpha = -L2*epsilon
        //cublasSaxpy(
        //    mBiasSurface->n_elems,
        //    alpha, 
        //    mBiasSurface->devPtr, 1,
        //    mSGD.mGradBiasVelocitySurface->devPtr, 1); // v = -L2*epsilon*w + v
        //alpha = -mHyperparameters.msgd.lr; // alpha = -epsilon
        //cublasSaxpy(
        //    mBiasSurface->n_elems,
        //    alpha, 
        //    mBiasGradSurface->devPtr, 1,
        //    mSGD.mGradBiasVelocitySurface->devPtr, 1); // v = -epsilon*grad + v
        //alpha = 1;
        //cublasSaxpy(
        //    mBiasSurface->n_elems,
        //    alpha, 
        //    mSGD.mGradBiasVelocitySurface->devPtr, 1,
        //    mBiasSurface->devPtr, 1); // w = v + w
    }

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        _printBias();
        _printFilter();
    }
}

void ConvBiasAct::saveParameters(const std::filesystem::path& dir, std::string_view NeuralNetworkName)
{
    mWeightsSurface->devToHostSync();
    mBiasSurface->devToHostSync();

    // TODO: save triaining stuff like grad and momentum too???

    auto weightsFilePath = dir / (std::string(NeuralNetworkName) + '_' + mName + std::to_string(mWeightsSurface->n_elems));
    auto biasFilePath = dir / (std::string(NeuralNetworkName) + '_' + mName + std::to_string(mBiasSurface->n_elems));

    if (mVerbosityLevel >= VERBOSITY::INFO) std::cout << std::format("Saving ConvBiasAct filter parameters to ", weightsFilePath.string()) << std::endl;

    std::ofstream filterFile(weightsFilePath, std::ios::binary);
    int64_t filterSize = mWeightsSurface->n_elems;
    int64_t filterElementSize = sizeof(mWeightsSurface->hostPtr[0]); // can be done better?

    filterFile.write(reinterpret_cast<char*>(&filterSize), sizeof(int64_t));
    filterFile.write(reinterpret_cast<char*>(&filterElementSize), sizeof(int64_t));
    
    filterFile.write(reinterpret_cast<char*>(mWeightsSurface->hostPtr), filterSize * filterElementSize);

    if (mVerbosityLevel >= VERBOSITY::INFO) std::cout << std::format("Saving Bias parameters to {}", biasFilePath.string()) << std::endl;

    std::ofstream biasFile(biasFilePath, std::ios::binary);
    int64_t biasSize = mBiasSurface->n_elems;
    int64_t biasElementSize = sizeof(mBiasSurface->hostPtr[0]); // can be done better? // should be equal to filter elements???

    biasFile.write(reinterpret_cast<char*>(&biasSize), sizeof(int64_t));
    biasFile.write(reinterpret_cast<char*>(&biasElementSize), sizeof(int64_t));

    biasFile.write(reinterpret_cast<char*>(mBiasSurface->hostPtr), biasSize * biasElementSize);
}

void ConvBiasAct::loadParameters(const std::filesystem::path& dir, std::string_view NeuralNetworkName)
{
    // TODO: load triaining stuff like grad and momentum too???

    auto filterFilePath = dir / (std::string(NeuralNetworkName) + '_' + mName + std::to_string(mWeightsSurface->n_elems));
    auto biasFilePath = dir / (std::string(NeuralNetworkName) + '_' + mName + std::to_string(mBiasSurface->n_elems));

    if (!std::filesystem::exists(filterFilePath) || !std::filesystem::exists(biasFilePath))
    {
        std::cout << std::format("No parameters to load. Expected file names are: {}, {}", filterFilePath.string(), biasFilePath.string()) << std::endl;
        return;
    }

    if (mVerbosityLevel >= VERBOSITY::INFO) std::cout << std::format("Loading filter parameters from {}", filterFilePath.string()) << std::endl;

    std::ifstream filterFile(filterFilePath, std::ios::binary);
    int64_t filterSize;
    int64_t filterElementSize;

    filterFile.read(reinterpret_cast<char*>(&filterSize), sizeof(int64_t)); // first parameter expected to be the number of elements
    filterFile.read(reinterpret_cast<char*>(&filterElementSize), sizeof(int64_t));
    assert(filterSize == mWeightsSurface->n_elems);
    assert(filterElementSize == sizeof(mWeightsSurface->hostPtr[0]));

    filterFile.read(reinterpret_cast<char*>(mWeightsSurface->hostPtr), filterSize * filterElementSize);

    if (mVerbosityLevel >= VERBOSITY::INFO) std::cout << std::format("Loading bias parameters from {}", biasFilePath.string()) << std::endl;

    std::ifstream biasFile(biasFilePath, std::ios::binary);
    int64_t biasSize;
    int64_t biasElementSize;

    biasFile.read(reinterpret_cast<char*>(&biasSize), sizeof(int64_t));
    biasFile.read(reinterpret_cast<char*>(&biasElementSize), sizeof(int64_t));
    assert(biasSize = mBiasSurface->n_elems);
    assert(biasElementSize = sizeof(mBiasSurface->hostPtr[0]));

    biasFile.read(reinterpret_cast<char*>(mBiasSurface->hostPtr), biasSize * biasElementSize);

    mWeightsSurface->hostToDevSync();
    mBiasSurface->hostToDevSync();
}

void ConvBiasAct::_setupBackPropagation()
{
    _setupActivationBackPropagation();
    _setupBiasBackPropagation();
    _setupDataBackPropagation();
    _setupFilterBackPropagation();
    _setupBackPropagationAlgorithms();

    //auto inputDim = mPreviousLayer->getOutputTensor().getDim();
    //auto outputDim = mOutputTensor->getDim();

    //Utils::checkCudnnError(cudnnCreateFilterDescriptor(&mFilterDesc));
    //Utils::checkCudnnError(cudnnSetFilter4dDescriptor(mFilterDesc,
    //    /*dataType=*/mDataType,
    //    /*format=*/mTensorFormat,
    //    /*out_channels=*/mFilterSize,
    //    /*in_channels=*/inputDim[1],
    //    /*kernel_height=*/mKernelSize,
    //    /*kernel_width=*/mKernelSize)); // most of convolution requires square kernels - change if needed

    //Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mBiasGradTensorDesc));
    //Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mBiasGradTensorDesc,
    //    mTensorFormat,
    //    mDataType,
    //    1, mFilterSize,
    //    1, 1));

    //Utils::checkCudnnError(cudnnCreateConvolutionDescriptor(&mConvDesc));
    //Utils::checkCudnnError(cudnnSetConvolution2dDescriptor(mConvDesc,
    //    /*pad_height=*/mPad[0],
    //    /*pad_width=*/mPad[1],
    //    /*vertical_stride=*/mConvStride[0],
    //    /*horizontal_stride=*/mConvStride[1],
    //    /*dilation_height=*/mDilation[0],
    //    /*dilation_width=*/mDilation[1],
    //    /*mode=*/mConvMode,
    //    /*computeType=*/mDataType));

    //Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mInputTensorDesc));
    //Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mGradTensorDesc));

    //Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mInputTensorDesc,
    //    /*format=*/mTensorFormat,
    //    /*dataType=*/mDataType,
    //    /*batch_size=*/inputDim[0],
    //    /*channels=*/inputDim[1],
    //    /*image_height=*/inputDim[2],
    //    /*image_width=*/inputDim[3]));

    //Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mGradTensorDesc,
    //    /*format=*/mTensorFormat,
    //    /*dataType=*/mDataType,
    //    /*batch_size=*/outputDim[0],
    //    /*channels=*/outputDim[1],
    //    /*image_height=*/outputDim[2],
    //    /*image_width=*/outputDim[3]));

    //constexpr int requestedAlgoCount = 1;
    //int returnedAlgoCount;

    //Utils::checkCudnnError(cudnnFindConvolutionBackwardDataAlgorithm(
    //    /*cudnnHandle_t                          */mHandle,
    //    /*const cudnnFilterDescriptor_t          */mFilterDesc,
    //    /*const cudnnTensorDescriptor_t          */mGradTensorDesc,
    //    /*const cudnnConvolutionDescriptor_t     */mConvDesc,
    //    /*const cudnnTensorDescriptor_t          */mInputTensorDesc,
    //    /*const int                              */requestedAlgoCount,
    //    /*int* */&returnedAlgoCount,
    //    /*cudnnConvolutionBwdDataAlgoPerf_t* */&mBwdDPerf));

    //Utils::checkCudnnError(cudnnFindConvolutionBackwardFilterAlgorithm(
    //    mHandle,
    //    mInputTensorDesc,
    //    mGradTensorDesc,
    //    mConvDesc,
    //    mFilterDesc,
    //    requestedAlgoCount,
    //    &returnedAlgoCount,
    //    &mBwdFPerf));

    //mDataGradWorkspaceSize = std::max(mBwdDPerf.memory, mBwdFPerf.memory);

    //if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << std::format("{} backpropagation descriptor setup completed. Workspace size: {}", mName, mDataGradWorkspaceSize) << std::endl;

    //if (mDataGradWorkspaceSize > 0)
    //{
    //    Utils::checkCudaError(cudaMalloc(&mDataGradWorkspacePtr, mDataGradWorkspaceSize));
    //}

    //mGradSurface = std::make_unique<Surface<float>>(mOutputSurface->n_elems, 0.0f);
    //mBiasGradSurface = std::make_unique<Surface<float>>(mBiasSurface->n_elems, 0.0f);
    //mWeightsGradSurface = std::make_unique<Surface<float>>(mWeightsSurface->n_elems, 0.0f);
    //mActivationGradSurface = std::make_unique<Surface<float>>(mOutputSurface->n_elems, 0.0f);

    if (mHyperparameters.updateType == Hyperparameters::UpdateType::mSGD)
    {
        mSGD.mGradBiasVelocitySurface = std::make_unique<Surface<float>>(mBiasSurface->n_elems, 0.0f);
        mSGD.mGradFilterVelocitySurface = std::make_unique<Surface<float>>(mWeightsSurface->n_elems, 0.0f);
    }

    //try
    //{
    //    //auto gradTensor = cudnn_frontend::TensorBuilder()
    //    //    .setAlignment(mOutputTensor->getAlignment())
    //    //    .setDataType(mDataType)
    //    //    .setDim(mOutputTensor->getDimCount(), mOutputTensor->getDim())
    //    //    .setStride(mOutputTensor->getDimCount(), mOutputTensor->getStride())
    //    //    .setId(generateTensorId())
    //    //    .build();
    //    auto gradTensor = Utils::createTensor(mOutputTensor->getDimCount(), mOutputTensor->getDim(), generateTensorId());
    //    //auto after_activation_tensor = cudnn_frontend::TensorBuilder()
    //    //    .setAlignment(mOutputTensor->getAlignment())
    //    //    .setDataType(mDataType)
    //    //    .setDim(mOutputTensor->getDimCount(), mOutputTensor->getDim())
    //    //    .setStride(mOutputTensor->getDimCount(), mOutputTensor->getStride())
    //    //    .setId(generateTensorId())
    //    //    .build();
    //    auto after_activation_tensor = Utils::createTensor(mOutputTensor->getDimCount(), mOutputTensor->getDim(), generateTensorId());
    //    // backwards relu
    //    auto actDesc = cudnn_frontend::PointWiseDescBuilder()
    //        .setMode(CUDNN_POINTWISE_RELU_BWD)
    //        .setComputeType(mDataType)
    //        .build();
    //    if (mVerbosityLevel >= VERBOSITY::DEBUG) std::cout << actDesc.describe() << std::endl;
    //    auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    //        .setdyDesc(gradTensor)
    //        .setxDesc(*mOutputTensor)
    //        .setdxDesc(after_activation_tensor)
    //        .setpwDesc(actDesc)
    //        .build();
    //    if (mVerbosityLevel >= VERBOSITY::DEBUG) std::cout << act_op.describe() << std::endl;
    //    std::vector<cudnn_frontend::Operation const*> ops;
    //    ops.emplace_back(&act_op);
    //    std::vector<void*> data_ptrs;
    //    data_ptrs.emplace_back(mGradSurface->devPtr);
    //    data_ptrs.emplace_back(mOutputSurface->devPtr);
    //    data_ptrs.emplace_back(mActivationGradSurface->devPtr);
    //    std::vector<int64_t> uids;
    //    uids.emplace_back(gradTensor.getId());
    //    uids.emplace_back(mOutputTensor->getId());
    //    uids.emplace_back(after_activation_tensor.getId());
    //    _setPlan(ops, data_ptrs, uids, mActivationGradPlan, mActivationGradVariantPack, mActivationGradWorkspaceSize, mActivationGradWorkspacePtr);
    //}
    //catch (cudnn_frontend::cudnnException& e) {
    //    std::cout << "[ERROR] Exception " << e.what() << std::endl;
    //    assert(false);
    //}
}

void ConvBiasAct::_setupActivationBackPropagation()
{
    mGradSurface = std::make_unique<Surface<float>>(mOutputSurface->n_elems, 0.0f);
    mActivationGradSurface = std::make_unique<Surface<float>>(mOutputSurface->n_elems, 0.0f);

    try
    {
        auto gradTensor = Utils::createTensor(mOutputTensor->getDimCount(), mOutputTensor->getDim(), generateTensorId());

        auto after_activation_tensor = Utils::createTensor(mOutputTensor->getDimCount(), mOutputTensor->getDim(), generateTensorId());

        // backwards relu
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_RELU_BWD)
            .setComputeType(mDataType)
            .build();
        if (mVerbosityLevel >= VERBOSITY::DEBUG) std::cout << actDesc.describe() << std::endl;

        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setdyDesc(gradTensor)
            .setxDesc(*mOutputTensor)
            .setdxDesc(after_activation_tensor)
            .setpwDesc(actDesc)
            .build();
        if (mVerbosityLevel >= VERBOSITY::DEBUG) std::cout << act_op.describe() << std::endl;
        std::vector<cudnn_frontend::Operation const*> ops;
        ops.emplace_back(&act_op);

        std::vector<void*> data_ptrs;
        data_ptrs.emplace_back(mGradSurface->devPtr);
        data_ptrs.emplace_back(mOutputSurface->devPtr);
        data_ptrs.emplace_back(mActivationGradSurface->devPtr);

        std::vector<int64_t> uids;
        uids.emplace_back(gradTensor.getId());
        uids.emplace_back(mOutputTensor->getId());
        uids.emplace_back(after_activation_tensor.getId());

        _setPlan(ops, data_ptrs, uids, mActivationGradPlan, mActivationGradVariantPack, mActivationGradWorkspaceSize, mActivationGradWorkspacePtr);
    }
    catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        assert(false);
    }
}

void ConvBiasAct::_setupBiasBackPropagation()
{
    mBiasGradSurface = std::make_unique<Surface<float>>(mBiasSurface->n_elems, 0.0f);
    Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mBiasGradTensorDesc));
    Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mBiasGradTensorDesc,
        mTensorFormat,
        mDataType,
        1, mFilterSize,
        1, 1));
}

void ConvBiasAct::_setupFilterBackPropagation()
{
    auto inputDim = mPreviousLayer->getOutputTensor().getDim();
    auto outputDim = mOutputTensor->getDim();
    mWeightsGradSurface = std::make_unique<Surface<float>>(mWeightsSurface->n_elems, 0.0f);

    Utils::checkCudnnError(cudnnCreateFilterDescriptor(&mFilterDesc));
    Utils::checkCudnnError(cudnnSetFilter4dDescriptor(mFilterDesc,
        /*dataType=*/mDataType,
        /*format=*/mTensorFormat,
        /*out_channels=*/mFilterSize,
        /*in_channels=*/inputDim[1],
        /*kernel_height=*/mKernelSize,
        /*kernel_width=*/mKernelSize)); // most of convolution requires square kernels - change if needed

    Utils::checkCudnnError(cudnnCreateConvolutionDescriptor(&mConvDesc));
    Utils::checkCudnnError(cudnnSetConvolution2dDescriptor(mConvDesc,
        /*pad_height=*/mPad[0],
        /*pad_width=*/mPad[1],
        /*vertical_stride=*/mConvStride[0],
        /*horizontal_stride=*/mConvStride[1],
        /*dilation_height=*/mDilation[0],
        /*dilation_width=*/mDilation[1],
        /*mode=*/mConvMode,
        /*computeType=*/mDataType));
}

void ConvBiasAct::_setupDataBackPropagation()
{
    auto inputDim = mPreviousLayer->getOutputTensor().getDim();
    auto outputDim = mOutputTensor->getDim();

    Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mInputTensorDesc));
    Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mGradTensorDesc));

    Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mInputTensorDesc,
        /*format=*/mTensorFormat,
        /*dataType=*/mDataType,
        /*batch_size=*/inputDim[0],
        /*channels=*/inputDim[1],
        /*image_height=*/inputDim[2],
        /*image_width=*/inputDim[3]));

    Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mGradTensorDesc,
        /*format=*/mTensorFormat,
        /*dataType=*/mDataType,
        /*batch_size=*/outputDim[0],
        /*channels=*/outputDim[1],
        /*image_height=*/outputDim[2],
        /*image_width=*/outputDim[3]));
}

void ConvBiasAct::_setupBackPropagationAlgorithms()
{
    constexpr int requestedAlgoCount = 1;
    int returnedAlgoCount;

    Utils::checkCudnnError(cudnnFindConvolutionBackwardDataAlgorithm(
        /*cudnnHandle_t                          */mHandle,
        /*const cudnnFilterDescriptor_t          */mFilterDesc,
        /*const cudnnTensorDescriptor_t          */mGradTensorDesc,
        /*const cudnnConvolutionDescriptor_t     */mConvDesc,
        /*const cudnnTensorDescriptor_t          */mInputTensorDesc,
        /*const int                              */requestedAlgoCount,
        /*int* */&returnedAlgoCount,
        /*cudnnConvolutionBwdDataAlgoPerf_t* */&mBwdDPerf));

    Utils::checkCudnnError(cudnnFindConvolutionBackwardFilterAlgorithm(
        mHandle,
        mInputTensorDesc,
        mGradTensorDesc,
        mConvDesc,
        mFilterDesc,
        requestedAlgoCount,
        &returnedAlgoCount,
        &mBwdFPerf));

    mDataGradWorkspaceSize = std::max(mBwdDPerf.memory, mBwdFPerf.memory);

    if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << std::format("{} backpropagation descriptor setup completed. Workspace size: {}", mName, mDataGradWorkspaceSize) << std::endl;

    if (mDataGradWorkspaceSize > 0)
    {
        Utils::checkCudaError(cudaMalloc(&mDataGradWorkspacePtr, mDataGradWorkspaceSize));
    }
}

void ConvBiasAct::_SGDUpdate()
{
    float alpha = -mLearningRate; // TODO: change name
    cublasSaxpy(
        mWeightsSurface->n_elems,
        alpha,
        mWeightsGradSurface->devPtr, 1,
        mWeightsSurface->devPtr, 1);
    cublasSaxpy(
        mBiasSurface->n_elems,
        alpha,
        mBiasGradSurface->devPtr, 1,
        mBiasSurface->devPtr, 1);
}

void ConvBiasAct::_miniBatchSGDUpdate()
{
    float alpha = 0;

    cublasSscal(
        mWeightsSurface->n_elems,
        mHyperparameters.msgd.momentum,
        mSGD.mGradFilterVelocitySurface->devPtr,
        1); // v = momentum * v

    alpha = -mHyperparameters.msgd.L2 * mHyperparameters.msgd.lr; // alpha = -L2*epsilon
    cublasSaxpy(
        mWeightsSurface->n_elems,
        alpha,
        mWeightsSurface->devPtr, 1,
        mSGD.mGradFilterVelocitySurface->devPtr, 1); // v = -L2*epsilon*w + v

    alpha = -mHyperparameters.msgd.lr; // alpha = -epsilon
    cublasSaxpy(
        mWeightsSurface->n_elems,
        alpha,
        mWeightsGradSurface->devPtr, 1,
        mSGD.mGradFilterVelocitySurface->devPtr, 1); // v = -epsilon*grad + v
    alpha = 1;
    cublasSaxpy(
        mWeightsSurface->n_elems,
        alpha,
        mSGD.mGradFilterVelocitySurface->devPtr, 1,
        mWeightsSurface->devPtr, 1); // w = v + w

    // bias
    cublasSscal(
        mBiasSurface->n_elems,
        mHyperparameters.msgd.momentum,
        mSGD.mGradBiasVelocitySurface->devPtr, 1); // v = momentum * v

    alpha = -mHyperparameters.msgd.L2 * mHyperparameters.msgd.lr; // alpha = -L2*epsilon
    cublasSaxpy(
        mBiasSurface->n_elems,
        alpha,
        mBiasSurface->devPtr, 1,
        mSGD.mGradBiasVelocitySurface->devPtr, 1); // v = -L2*epsilon*w + v

    alpha = -mHyperparameters.msgd.lr; // alpha = -epsilon
    cublasSaxpy(
        mBiasSurface->n_elems,
        alpha,
        mBiasGradSurface->devPtr, 1,
        mSGD.mGradBiasVelocitySurface->devPtr, 1); // v = -epsilon*grad + v

    alpha = 1;
    cublasSaxpy(
        mBiasSurface->n_elems,
        alpha,
        mSGD.mGradBiasVelocitySurface->devPtr, 1,
        mBiasSurface->devPtr, 1); // w = v + w
}


// TODO: move to Surface
void ConvBiasAct::_printBias()
{
    std::cout << std::format("{} bias:", mName) << std::endl;
    mBiasSurface->devToHostSync();

    for (int64_t b = 0; b < mBiasSurface->n_elems; ++b)
    {
        std::cout << std::format("{} ", mBiasSurface->hostPtr[b]);
    }

    std::cout << std::endl;
}

void ConvBiasAct::_printFilter()
{
    std::cout << std::format("{} filter with {} elements:", mName, mWeightsSurface->n_elems) << std::endl;
    mWeightsSurface->devToHostSync();

    for (int64_t w = 0; w < std::min(mWeightsSurface->n_elems, 100ll); ++w)
    {
        std::cout << std::format("{} ", mWeightsSurface->hostPtr[w]);
    }

    std::cout << std::endl;
}

void ConvBiasAct::_printActivationGrad()
{
    std::cout << std::format("{} mActivationGradSurface:", mName) << std::endl;
    mActivationGradSurface->devToHostSync();

    for (int64_t b = 0; b < mActivationGradSurface->n_elems; ++b)
    {
        std::cout << std::format("{} ", mActivationGradSurface->hostPtr[b]);
    }

    std::cout << std::endl;
}

void ConvBiasAct::_printBiasGrad()
{
    std::cout << std::format("{} mBiasGradSurface:", mName) << std::endl;
    mBiasGradSurface->devToHostSync();

    for (int64_t b = 0; b < mBiasGradSurface->n_elems; ++b)
    {
        std::cout << std::format("{} ", mBiasGradSurface->hostPtr[b]);
    }

    std::cout << std::endl;
}

void ConvBiasAct::_printFilterGrad()
{
    std::cout << std::format("{} mWeightsGradSurface:", mName) << std::endl;
    mWeightsGradSurface->devToHostSync();

    for (int64_t b = 0; b < std::min(mWeightsGradSurface->n_elems, 1000ll); ++b)
    {
        std::cout << std::format("{} ", mWeightsGradSurface->hostPtr[b]);
    }

    std::cout << std::endl;
}
