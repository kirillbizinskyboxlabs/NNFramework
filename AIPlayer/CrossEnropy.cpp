#include "CrossEnropy.h"

import <format>;
#include <iomanip>

CrossEntropy::CrossEntropy(cudnnHandle_t& handle, 
    Layer* previousLayer, 
    const Hyperparameters& hyperparameters,
    std::string name,
    VERBOSITY verbosity)
    : Layer(handle, previousLayer, hyperparameters, std::move(name), verbosity)
    , mGradWorkspaceSize(0)
    , mGradWorkspacePtr(nullptr)
{
    if (mVerbosityLevel >= VERBOSITY::INFO)
    {
        std::cout << std::format("Creating Cross Entropy Loss Layer") << std::endl;
    }

    _initLoss();
    _initGrad();
}

CrossEntropy::~CrossEntropy()
{
    if (mGradWorkspacePtr)
    {
        cudaFree(mGradWorkspacePtr);
        mGradWorkspacePtr = nullptr;
    }
}

void CrossEntropy::printOutput()
{
    auto print = [batchSize = mBatchSize, numClasses = mNumClasses](float* hostPtr)
    {
        for (int64_t b = 0; b < std::min(batchSize, 10ll); ++b)
        {
            //float* outputHostPtr = mPreviousLayer->getOutputSurface().hostPtr;
            auto beg = hostPtr + numClasses * b;
            auto end = hostPtr + numClasses * (b + 1);
            auto it = std::max_element(beg, end);
            std::cout << std::distance(beg, it) << " ";
        }
        std::cout << std::endl;
    };

    std::cout << "Labels and NN output:" << std::endl;

    mLabelSurface->devToHostSync();
    print(mLabelSurface->hostPtr);

    mPreviousLayer->getOutputSurface().devToHostSync();
    print(mPreviousLayer->getOutputSurface().hostPtr);
}

void CrossEntropy::printLoss()
{
    //mLossSurface->devToHostSync();

    auto loss = getLoss();

    std::cout << std::format("Loss: {}", loss) << std::endl;
}

void CrossEntropy::printGrad()
{
    mPreviousLayer->getGradSurface().devToHostSync();

    constexpr int precision = 8;

    std::cout << "Cross Entropy derivative:" << std::endl;

    for (int64_t b = 0; b < mBatchSize; ++b)
    {
        for (int64_t i = 0; i < mNumClasses; ++i)
        {
            std::cout << std::setprecision(precision) << mPreviousLayer->getGradSurface().hostPtr[b * mNumClasses + i] << " ";
        }
        std::cout << std::endl;
    }
}

float CrossEntropy::getLoss()
{
    mLossSurface->devToHostSync();

    return -1 * mLossSurface->hostPtr[0] / mBatchSize;
}

void CrossEntropy::propagateBackward()
{
    if (mVerbosityLevel >= VERBOSITY::DEBUG) std::cout << "Back Propagating on CrossEntropy" << std::endl;
    calculateGrad();
}

void CrossEntropy::calculateLoss()
{
    //TODO: change naming to Loss specific
    if (!mForwardPropagationPlan || !mForwardPropagationVariantPack)
    {
        throw;
    }

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        std::cout << "calculateLoss" << std::endl;
    }

    Utils::checkCudnnError(cudnnBackendExecute(mHandle, mForwardPropagationPlan->get_raw_desc(), mForwardPropagationVariantPack->get_raw_desc()));
}

void CrossEntropy::calculateGrad()
{
    // deduplicate?
    if (!mGradPlan || !mGradVariantPack)
    {
        throw;
    }

    if (mVerbosityLevel >= VERBOSITY::DEBUG)
    {
        std::cout << "calculateGrad" << std::endl;
    }

    Utils::checkCudnnError(cudnnBackendExecute(mHandle, mGradPlan->get_raw_desc(), mGradVariantPack->get_raw_desc()));
}

void CrossEntropy::setLabel(std::span<uint8_t> labels)
{
    assert(labels.size() == mBatchSize);

    for (int64_t b = 0; b < mBatchSize; ++b)
    {
        for (int64_t i = 0; i < mNumClasses; ++i)
        {
            if (labels[b] == i)
            {
                mLabelSurface->hostPtr[b * mNumClasses + i] = 1;
            }
            else
            {
                mLabelSurface->hostPtr[b * mNumClasses + i] = 0;
            }
        }
    }

    mLabelSurface->hostToDevSync();
}

float* CrossEntropy::getLabelDataPtr()
{
    assert(mLabelSurface);
    return mLabelSurface->hostPtr;
}

void CrossEntropy::syncLabel()
{
    mLabelSurface->hostToDevSync();
}

void CrossEntropy::_initLoss()
{
    //TODO: rename
    auto inputTensor = _flattenTensor(mPreviousLayer->getOutputTensor());
    auto inputDim = inputTensor.getDim();
    mBatchSize = inputDim[0];
    assert(inputDim[1] == 1); //sanity check // TODO: multidimensional cross-entropy?
    mNumClasses = inputDim[2]; //we expect it to be flatten // TODO: sanity check?
    //mNumClasses = inputDim[1];
    //assert(inputDim[2] == 1); // TODO: rethink
    //assert(inputDim[3] == 1);
    const int64_t reshapedInputDim[] = { mBatchSize, 1, mNumClasses };
    const int64_t crossEntropyLabelDim[] = { mBatchSize, mNumClasses, 1 }; // we need it to be orthogonal to softmax output, so that we can have a working dot product
    const int64_t crossEntropyLabelStride[] = { mNumClasses, 1, 1 }; // does that makes sense?
    const int64_t matmulDim[] = { mBatchSize, 1, 1 }; // magic numbers are justified here?
    const int64_t matmulStride[] = { 1,1,1 };

    mProductSurface = std::make_unique<Surface<float>>(matmulDim[0] * matmulDim[1] * matmulDim[2], 0.0f);

    //int64_t labelStride[] = { mNumClasses, mNumClasses, 1 }; // does this?

    int64_t lossDim[] = { 1, 1, 1 }; // I think it should be that since we need to have an average across dims including the batch
    int64_t lossStride[] = { 1,1,1 }; // not sure if this makes sense
    mLossSurface = std::make_unique<Surface<float>>(lossDim[0] * lossDim[1] * lossDim[2], 0.0f);

    mLabelSurface = std::make_unique<Surface<float>>(mBatchSize * mNumClasses, 0.0f);

    // HYPERPARAMETERS
    constexpr int64_t alignment = 16;
    const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    constexpr int64_t nbDims = 3;

    using namespace cudnn_frontend;

    if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << inputTensor.describe() << std::endl;

    try
    {
        auto crossEntropyLabelTensor = TensorBuilder()
            .setDataType(dataType)
            .setAlignment(alignment) // this needs to be a function
            .setDim(nbDims, crossEntropyLabelDim)
            .setStride(nbDims, crossEntropyLabelStride)
            .setId(generateTensorId())
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << crossEntropyLabelTensor.describe() << std::endl;

        auto afterProductTensor = TensorBuilder()
            .setDataType(dataType)
            .setAlignment(alignment)
            .setDim(nbDims, matmulDim)
            .setStride(nbDims, matmulStride)
            .setId(generateTensorId())
            //.setVirtual(true) // apparently it cannot be virtual 
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << afterProductTensor.describe() << std::endl;

        auto afterLogTensor = TensorBuilder()
            .setDataType(dataType)
            .setAlignment(alignment)
            .setDim(nbDims, matmulDim)
            .setStride(nbDims, matmulStride)
            .setId(generateTensorId())
            .setVirtual(true)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << afterLogTensor.describe() << std::endl;

        // the output
        auto lossTensor = TensorBuilder()
            .setDataType(dataType)
            .setAlignment(alignment)
            .setDim(nbDims, lossDim)
            .setStride(nbDims, lossStride)
            .setId(generateTensorId())
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << lossTensor.describe() << std::endl;

        // loss ops

        auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
        // Create a matmul Node
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "Matmul" << std::endl;
        auto matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
            .setaMatDesc(inputTensor)
            .setbMatDesc(crossEntropyLabelTensor)
            .setcMatDesc(afterProductTensor)
            .setmatmulDesc(matmulDesc)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << matmul_op.describe() << std::endl;

        auto pwLogDesc = PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_LOG)
            .setComputeType(dataType)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << pwLogDesc.describe() << std::endl;

        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "pwLog_op" << std::endl;
        auto pwLog_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(matmul_op.getOutputTensor())
            .setyDesc(afterLogTensor)
            .setpwDesc(pwLogDesc)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << pwLog_op.describe() << std::endl;

        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "sumReductionDesc" << std::endl;
        auto sumReductionDesc = ReductionDescBuilder()
            .setComputeType(dataType)
            .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << sumReductionDesc.describe() << std::endl;

        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << "sumReduction_op" << std::endl;
        auto sumReduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
            .setxDesc(pwLog_op.getOutputTensor())
            .setyDesc(lossTensor)
            .setreductionDesc(sumReductionDesc)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << sumReduction_op.describe() << std::endl;

        std::vector<cudnn_frontend::Operation const*> ops = { &matmul_op, &pwLog_op, &sumReduction_op };

        std::vector<void*> data_ptrs;
        data_ptrs.emplace_back(mPreviousLayer->getOutputSurface().devPtr);
        data_ptrs.emplace_back(mLabelSurface->devPtr);
        data_ptrs.emplace_back(mProductSurface->devPtr);
        data_ptrs.emplace_back(mLossSurface->devPtr);

        std::vector<int64_t> uids;
        //uids.emplace_back(mPreviousLayer->getOutputTensor().getId());
        uids.emplace_back(inputTensor.getId());
        uids.emplace_back(crossEntropyLabelTensor.getId());
        uids.emplace_back(afterProductTensor.getId());
        uids.emplace_back(lossTensor.getId());

        _setForwardPropagationPlan(ops, data_ptrs, uids);
    }
    catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
    }
}

void CrossEntropy::_initGrad()
{
    // TODO: rename
    auto inputTensor = _flattenTensor(mPreviousLayer->getOutputTensor());
    // TODO: Proper defaults
    constexpr int64_t alignment = 16;
    const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    constexpr int64_t nbDims = 3;

    using namespace cudnn_frontend;

    auto inputDim = inputTensor.getDim();
    assert(inputTensor.getDimCount() == nbDims); // sanity check, remove if ever allow different dimensiality
    int64_t gradDim[] = { inputDim[0], inputDim[1], inputDim[2] };
    int64_t gradStride[] = { inputDim[1] * inputDim[2], inputDim[2], 1};

    if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << inputTensor.describe() << std::endl;
    try
    {
        auto labelTensor = TensorBuilder()
            .setDataType(dataType)
            .setAlignment(alignment)
            .setId(generateTensorId())
            .setDim(nbDims, gradDim)
            .setStride(nbDims, gradStride)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << labelTensor.describe() << std::endl;

        auto gradTensor = TensorBuilder()
            .setDataType(dataType)
            .setAlignment(alignment)
            .setId(generateTensorId())
            .setDim(nbDims, gradDim)
            .setStride(nbDims, gradStride)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << gradTensor.describe() << std::endl;

        auto subDesc = PointWiseDescBuilder()
            .setComputeType(dataType)
            .setMode(CUDNN_POINTWISE_SUB)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << subDesc.describe() << std::endl;

        auto pw_sub_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(inputTensor)
            .setbDesc(labelTensor)
            .setyDesc(gradTensor)
            .setpwDesc(subDesc)
            .build();
        if (mVerbosityLevel >= VERBOSITY::REACH_INFO) std::cout << pw_sub_op.describe() << std::endl;

        // TODO: this block needs to be a function...
        std::vector<cudnn_frontend::Operation const*> ops = { &pw_sub_op };

        std::vector<void*> data_ptrs;
        data_ptrs.emplace_back(mPreviousLayer->getOutputSurface().devPtr);
        data_ptrs.emplace_back(mLabelSurface->devPtr);
        data_ptrs.emplace_back(mPreviousLayer->getGradSurface().devPtr);

        std::vector<int64_t> uids;
        uids.emplace_back(inputTensor.getId());
        uids.emplace_back(labelTensor.getId());
        uids.emplace_back(gradTensor.getId());

        _setPlan(ops, data_ptrs, uids, mGradPlan, mGradVariantPack, mGradWorkspaceSize, mGradWorkspacePtr);
    }
    catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
    }
}

cudnn_frontend::Tensor CrossEntropy::_flattenTensor(cudnn_frontend::Tensor& tensor)
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
        const int64_t FCstride[3] = { flattenTensorDim[1] * flattenTensorDim[2], 1, 1 };
        //Helpers::generateStrides(flattenTensorDim, FCstride, 3, tensorFormat);
        // 
        // RVO
        return cudnn_frontend::TensorBuilder()
            .setDim(3, flattenTensorDim)
            .setStride(3, FCstride)
            .setId(generateTensorId())
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
            .setId(generateTensorId())
            .setAlignment(tensor.getAlignment())
            .setDataType(static_cast<cudnnDataType_t>(tensor.getDataType()))
            .build();
    }
}