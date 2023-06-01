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
    bool verbose,
    std::string name)
    : Layer(handle, previousLayer, hyperparameters, verbose, std::move(name))
    , mLearningRate(learningRate)
    , mNeedDataGrad(needDataGrad)
{
    // Defaults. TODO: Move to the appropriate place. Some of these are hyperparameters. Some are necessary constants. Bad place.
    constexpr int64_t alignment = 16; //16B to make Tensor cores work
    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    constexpr int convDim = 2;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    constexpr int64_t nbDims = 4;

    auto inputDim = mPreviousLayer->getOutputTensor().getDim();

    const int64_t wTensorDim[] = { filterSize, inputDim[1], kernelSize, kernelSize }; // filter
    int64_t yTensorDim[] = { 0, 0, 0, 0 }; // Computed Below
    const int64_t padA[] = { convPad, convPad };
    const int64_t dilationA[] = { 1, 1 }; // TODO: make proper defaults
    const int64_t convstrideA[] = { 1, 1 };
    const int64_t bTensorDim[] = { 1, wTensorDim[0], 1, 1 };  // bias

    yTensorDim[0] = inputDim[0];
    yTensorDim[1] = wTensorDim[0];
    for (int dim = 0; dim < 2; dim++) {
        yTensorDim[dim + 2] = Utils::getFwdConvOutputDim(inputDim[dim + 2], padA[dim], wTensorDim[dim + 2], convstrideA[dim], dilationA[dim]);
    }

    if (verbose)
    {
        std::cout << std::format("====DIMENSIONS====") << std::endl;
        std::cout << std::format("input dims are {}, {}, {}, {}", inputDim[0], inputDim[1], inputDim[2], inputDim[3]) << std::endl;
        std::cout << std::format("filter dims are {}, {}, {}, {}", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]) << std::endl;
        std::cout << std::format("output dims are {}, {}, {}, {}", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]) << std::endl;
    }

    int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

    constexpr float biasStartValue = 0.1f;

    mWeightsSurface = std::make_unique<Surface<float>>(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3]);
    mBiasSurface = std::make_unique<Surface<float>>(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], biasStartValue);
    mOutputSurface = std::make_unique<Surface<float>>(Ysize, 0.0f);

    // Xavier filter values
    std::random_device dev;
    std::mt19937 gen(dev());
    const float deviation = sqrt(3.0f / (inputDim[1] * kernelSize * kernelSize)); // input channels * KS^2
    //const float deviation = 1.0f / (inputDim[1] * kernelSize * kernelSize); // input channels * KS^2
    std::uniform_real_distribution<float> distribution(-deviation, deviation);

    for (int64_t i = 0; i < mWeightsSurface->n_elems; ++i)
    {
        mWeightsSurface->hostPtr[i] = distribution(gen);
    }

    mWeightsSurface->hostToDevSync();

    try
    {
        int64_t stride[nbDims];
        Utils::generateStrides(wTensorDim, stride, nbDims, tensorFormat);
        auto wTensor = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, wTensorDim)
            .setStride(nbDims, stride)
            .setId(generateTensorId())
            .setAlignment(alignment)
            .setDataType(dataType)
            .build();

        Utils::generateStrides(bTensorDim, stride, nbDims, tensorFormat);
        auto bTensor = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, bTensorDim)
            .setStride(nbDims, stride)
            .setId(generateTensorId())
            .setAlignment(alignment)
            .setDataType(dataType)
            .build();

        Utils::generateStrides(yTensorDim, stride, nbDims, tensorFormat);
        auto afterConvTensor = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, yTensorDim)
            .setStride(nbDims, stride)
            .setId(generateTensorId())  // after conv
            .setAlignment(alignment)
            .setVirtual()
            .setDataType(dataType)
            .build();

        auto afterBiasTensor = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, yTensorDim)
            .setStride(nbDims, stride)
            .setId(generateTensorId())  // after bias
            .setAlignment(alignment)
            .setVirtual()
            .setDataType(dataType)
            .build();

        mOutputTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
            .setDim(nbDims, yTensorDim)
            .setStride(nbDims, stride)
            .setId(generateTensorId())  // after relu
            .setAlignment(alignment)
            .setDataType(dataType)
            .build());

        if (mVerbose)
        {
            std::cout << mPreviousLayer->getOutputTensor().describe() << std::endl;
            std::cout << wTensor.describe() << std::endl;
            std::cout << bTensor.describe() << std::endl;
            std::cout << afterConvTensor.describe() << std::endl;
            std::cout << mOutputTensor->describe() << std::endl;
        }

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
            .setComputeType(dataType)
            .setMathMode(CUDNN_CROSS_CORRELATION)
            .setSpatialDimCount(convDim)
            .setSpatialStride(convDim, convstrideA)
            .setPrePadding(convDim, padA)
            .setPostPadding(convDim, padA)
            .setDilation(convDim, dilationA)
            .build();
        if (mVerbose) std::cout << convDesc.describe() << std::endl;

        // Define the bias descriptor
        auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_ADD)
            .setComputeType(dataType)
            .build();
        if (mVerbose) std::cout << biasDesc.describe() << std::endl;

        // Define the activation descriptor
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_RELU_FWD)//CUDNN_POINTWISE_SIGMOID_FWD
            .setComputeType(dataType)
            .build();
        if (mVerbose) std::cout << actDesc.describe() << std::endl;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            .setxDesc(mPreviousLayer->getOutputTensor())
            .setwDesc(wTensor)
            .setyDesc(afterConvTensor) 
            .setcDesc(convDesc)
            .setAlpha(alpha)
            .setBeta(beta)
            .build();
        if (mVerbose) std::cout << conv_op.describe() << std::endl;

        // Create a Bias Node.
        auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(conv_op.getOutputTensor())
            .setbDesc(bTensor)
            .setyDesc(afterBiasTensor)
            .setpwDesc(biasDesc)
            .build();
        if (mVerbose) std::cout << bias_op.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(bias_op.getOutputTensor())
            .setyDesc(*mOutputTensor)
            .setpwDesc(actDesc)
            .build();
        if (mVerbose) std::cout << act_op.describe() << std::endl;

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
        uids.emplace_back(wTensor.getId());
        uids.emplace_back(bTensor.getId());
        uids.emplace_back(mOutputTensor->getId());

        _setPlan(ops, data_ptrs, uids, mForwardPropagationPlan, mForwardPropagationVariantPack, mForwardPropagationWorkspaceSize, mForwardPropagationWorkspacePtr);
    }
    catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
    }

    if (training)
    {
        mGradSurface = std::make_unique<Surface<float>>(Ysize, 0.0f);

        Utils::checkCudnnError(cudnnCreateFilterDescriptor(&mFilterDesc));
        Utils::checkCudnnError(cudnnSetFilter4dDescriptor(mFilterDesc,
            /*dataType=*/dataType,
            /*format=*/tensorFormat,
            /*out_channels=*/filterSize,
            /*in_channels=*/inputDim[1],
            /*kernel_height=*/kernelSize,
            /*kernel_width=*/kernelSize)); // most of convolution requires square kernels - change if needed

        Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mBiasGradTensorDesc));
        Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mBiasGradTensorDesc,
            tensorFormat,
            dataType,
            1, filterSize,
            1, 1));

        Utils::checkCudnnError(cudnnCreateConvolutionDescriptor(&mConvDesc));
        Utils::checkCudnnError(cudnnSetConvolution2dDescriptor(mConvDesc,
            /*pad_height=*/convPad,
            /*pad_width=*/convPad,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/1,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/dataType));

        //cudnnGetConvolutionBackwardDataAlgorithm_v7
        //cudnnConvolutionBwdDataPreference_t bwdDPref;
        //cudnnConvolutionBwdFilterPreference_t bwdFPref;

        Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mInputTensorDesc));
        Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mGradTensorDesc));

        Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mInputTensorDesc,
            /*format=*/tensorFormat,
            /*dataType=*/dataType,
            /*batch_size=*/inputDim[0],
            /*channels=*/inputDim[1],
            /*image_height=*/inputDim[2],
            /*image_width=*/inputDim[3]));

        Utils::checkCudnnError(cudnnSetTensor4dDescriptor(mGradTensorDesc,
            /*format=*/tensorFormat,
            /*dataType=*/dataType,
            /*batch_size=*/inputDim[0],
            /*channels=*/yTensorDim[1],
            /*image_height=*/yTensorDim[2],
            /*image_width=*/yTensorDim[3]));
        
        const int requestedAlgoCount = 1;
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

        if (mVerbose) std::cout << std::format("{} backpropagation descriptor setup completed. Workspace size: {}", mName, mDataGradWorkspaceSize) << std::endl;

        if (mDataGradWorkspaceSize > 0)
        {
            Utils::checkCudaError(cudaMalloc(&mDataGradWorkspacePtr, mDataGradWorkspaceSize));
        }

        //Utils::checkCudnnError(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        //    mHandle, 
        //    src->tensor, 
        //    tensor, 
        //    mConvDesc,
        //    mFilterDesc,
        //    bwF_algo, 
        //    &workspace_bytes));

        //nn->m_workspaceSize = std::max(nn->m_workspaceSize, workspace_bytes);

        //Utils::checkCudnnError(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        //    mHandle, 
        //    filterDesc, 
        //    tensor, 
        //    desc, 
        //    src->tensor,
        //    bwdDPref,
        //    /*memoryLimitInBytes=*/memory_limit,
        //    &bwD_algo));

        //Utils::checkCudnnError(cudnnGetConvolutionBackwardDataWorkspaceSize(
        //    mHandle, 
        //    mFilterDesc,
        //    tensor, 
        //    mConvDesc,
        //    src->tensor,
        //    bwdDPerf.algo,
        //    &workspace_bytes));

        _setupBackPropagation(needDataGrad);
    }
}

void ConvBiasAct::propagateBackward()
{
    Utils::checkCudnnError(cudnnBackendExecute(mHandle, mActivationGradPlan->get_raw_desc(), mActivationGradVariantPack->get_raw_desc()));

    if (mVerbose)
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

    if (mVerbose)
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
    if (mHyperparameters.updateType == Hyperparameters::UpdateType::SGD) // SGD
    {
        float alpha = -mLearningRate; // TODO: change name
        cublasSaxpy(
            //nn->cublasHandle, 
            mWeightsSurface->n_elems,
            alpha, 
            mWeightsGradSurface->devPtr, 1,
            mWeightsSurface->devPtr, 1);
        cublasSaxpy(
            //nn->cublasHandle, 
            //static_cast<int>(bias_value.size()),
            mBiasSurface->n_elems,
            alpha, 
            mBiasGradSurface->devPtr, 1,
            mBiasSurface->devPtr, 1);
    }
    else if (mHyperparameters.updateType == Hyperparameters::UpdateType::mSGD)
    {
        //std::cout << out_channels << " backprop mSGD\n";
        //show_some_data(filter_data, 10);
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
        //show_some_data(filter_data, 10);

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

    if (mVerbose)
    {
        _printBias();
        _printFilter();
    }
}

void ConvBiasAct::_setupBackPropagation(bool needDataGrad)
{
    mGradSurface = std::make_unique<Surface<float>>(mOutputSurface->n_elems, 0.0f);
    mBiasGradSurface = std::make_unique<Surface<float>>(mBiasSurface->n_elems, 0.0f);
    mWeightsGradSurface = std::make_unique<Surface<float>>(mWeightsSurface->n_elems, 0.0f);
    mActivationGradSurface = std::make_unique<Surface<float>>(mOutputSurface->n_elems, 0.0f);

    if (mHyperparameters.updateType == Hyperparameters::UpdateType::mSGD)
    {
        mSGD.mGradBiasVelocitySurface = std::make_unique<Surface<float>>(mBiasSurface->n_elems, 0.0f);
        mSGD.mGradFilterVelocitySurface = std::make_unique<Surface<float>>(mWeightsSurface->n_elems, 0.0f);
    }

    try
    {
        auto gradTensor = cudnn_frontend::TensorBuilder()
            .setAlignment(mOutputTensor->getAlignment())
            .setDataType(CUDNN_DATA_FLOAT)
            .setDim(mOutputTensor->getDimCount(), mOutputTensor->getDim())
            .setStride(mOutputTensor->getDimCount(), mOutputTensor->getStride())
            .setId(generateTensorId())
            .build();

        auto after_activation_tensor = cudnn_frontend::TensorBuilder()
            .setAlignment(mOutputTensor->getAlignment())
            .setDataType(CUDNN_DATA_FLOAT)
            .setDim(mOutputTensor->getDimCount(), mOutputTensor->getDim())
            .setStride(mOutputTensor->getDimCount(), mOutputTensor->getStride())
            .setId(generateTensorId())
            .build();

        // backwards relu
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_RELU_BWD)
            .setComputeType(CUDNN_DATA_FLOAT)
            .build();
        if (mVerbose) std::cout << actDesc.describe() << std::endl;

        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setdyDesc(gradTensor)
            .setxDesc(*mOutputTensor)
            .setdxDesc(after_activation_tensor)
            .setpwDesc(actDesc)
            .build();
        if (mVerbose) std::cout << act_op.describe() << std::endl;
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
    ////std::cout << "Creating reduce tensor" << std::endl;
    ////cudnnReduceTensorDescriptor_t reduceTensorDesc;
    //Utils::checkCudnnError(cudnnCreateReduceTensorDescriptor(&mReduceTensorDesc));

    ////std::cout << "Setting reduce tensor" << std::endl;
    //Utils::checkCudnnError(cudnnSetReduceTensorDescriptor(
    //    /*cudnnReduceTensorDescriptor_t   */mReduceTensorDesc,
    //    /*cudnnReduceTensorOp_t           */CUDNN_REDUCE_TENSOR_ADD,
    //    /*cudnnDataType_t                 */CUDNN_DATA_FLOAT,
    //    /*cudnnNanPropagation_t           */CUDNN_NOT_PROPAGATE_NAN,
    //    /*cudnnReduceTensorIndices_t      */CUDNN_REDUCE_TENSOR_NO_INDICES,
    //    /*cudnnIndicesType_t              */CUDNN_64BIT_INDICES));

    ////std::cout << "Creating grad tensors" << std::endl;
    //cudnnTensorDescriptor_t mGradTensorDesc, mBiasGradTensorDesc;
    //Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mGradTensorDesc));
    //Utils::checkCudnnError(cudnnCreateTensorDescriptor(&mBiasGradTensorDesc));

    //// TODO: Proper defaults
    //constexpr int nbDims = 3;

    ////auto& inputTensor = prevLayer.getOutputTensor();

    ////assert(nbDims == inputTensor.getDimCount());

    //auto gradDim = mOutputTensor->getDim();
    //auto outputStride = mOutputTensor->getStride();
    //int mGradDims[] = { static_cast<int>(gradDim[0]), static_cast<int>(gradDim[1]), static_cast<int>(gradDim[2]), static_cast<int>(gradDim[3]) };
    ////int gradStride[] = { mGradDims[1] * mGradDims[2] * mGradDims[3] , mGradDims[2] * mGradDims[3], mGradDims[3], 1 };
    //int gradStride[] = { outputStride[1], outputStride[2], outputStride[3], outputStride[4] };

    //int mBiasGradDims[] = { 1, mGradDims[1], mGradDims[2] }; // reduced over batch
    ////int mBiasGradStride[] = { gradDim[1] * gradDim[2] , gradDim[2], 1 };

    ////cudnnReduc

    ////std::cout << "Setting grad tensors" << std::endl;
    //Utils::checkCudnnError(cudnnSetTensorNdDescriptor(mGradTensorDesc,
    //    dataType,
    //    nbDims,
    //    mGradDims,
    //    gradStride));
    //Utils::checkCudnnError(cudnnSetTensorNdDescriptor(mBiasGradTensorDesc,
    //    dataType,
    //    nbDims,
    //    mBiasGradDims,
    //    gradStride));

    ////std::cout << "Getting workspace size" << std::endl;
    //// Get workspace size for reduce operation
    //size_t mBiasGradWorkspaceSize;
    //Utils::checkCudnnError(cudnnGetReductionWorkspaceSize(mHandle, mReduceTensorDesc, mGradTensorDesc, mBiasGradTensorDesc, &mBiasGradWorkspaceSize));
    ////std::cout << std::format("Reduction op will need {} size workspace", mBiasGradWorkspaceSize) << std::endl;

    //// Allocate memory for workspace
    //void* mBiasGradWorkspacePtr;
    //cudaMalloc(&mBiasGradWorkspacePtr, mBiasGradWorkspaceSize);


}

void ConvBiasAct::_setupWeightBackPropagation()
{
}

void ConvBiasAct::_setupDataBackPropagation()
{
}

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
