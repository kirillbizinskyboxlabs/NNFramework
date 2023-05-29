#include "LenetLayers.h"
//#include "LeNetHelpers.h"
#include <cublas.h>

import <iostream>;
import <format>;
import <exception>;

namespace LeNet
{
    constexpr float epsilon = 0.01; // learning rate
    constexpr int64_t alignment = 16; //16B to make Tensor cores work
    constexpr cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    constexpr cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    constexpr int convDim = 2;
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    void display(float* image, size_t size)
    {
        for (size_t r = 0; r < size; ++r)
        {
            for (size_t c = 0; c < size; ++c)
            {
                std::cout << std::setw(3) << std::setprecision(3) << image[r * size + c] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "\n";
    }

    void display_flat(float* image, size_t size)
    {
        for (size_t i = 0; i < size; ++i)
        {
            std::cout << std::setprecision(8) << image[i] << " ";
        }

        std::cout << "\n";
    }

    Layer::Layer(cudnnHandle_t& handle, bool verbose, std::string name)
        : mHandle(handle)
        , mVerbose(verbose)
        , mName(std::move(name))
    {
    }

    Layer::~Layer()
    {
        if (workspace_size > 0)
        {
            checkCudaErr(cudaFree(workspace_ptr));
        }
    }

    int Layer::executeInference()
    {
        if (!plan || !variantPack)
        {
            return 1; // not initialized
        }

        if (mVerbose)
        {
            std::cout << std::format("Executing inference on {}", mName.c_str()) << std::endl;
        }

        cudnnStatus_t status;
        status = cudnnBackendExecute(mHandle, plan->get_raw_desc(), variantPack->get_raw_desc());
        if (status != CUDNN_STATUS_SUCCESS)
        {
            std::cout << cudnnGetErrorString(status) << std::endl;
            return 2; // something went wrong during execution. TODO: return status instead???
        }

        // OK
        return 0;
    }

    cudnn_frontend::Tensor& Layer::getOutputTensor() const
    {
        assert(yTensor); // Assert it's initialized
        return *yTensor;
    }

    Helpers::Surface<float>& Layer::getOutputSurface() const
    {
        assert(Y);
        return *Y;
    }

    Helpers::Surface<float>& Layer::getGradSurface() const
    {
        assert(mGradSurface);
        return *mGradSurface;
    }

    void Layer::printOutput()
    {
        if (!Y)
        {
            throw;
        }

        cudaDeviceSynchronize();
        checkCudaErr(cudaMemcpy(Y->hostPtr, Y->devPtr, (size_t)(sizeof(Y->hostPtr[0]) * Y->n_elems), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        auto yDim = yTensor->getDim();
        for (size_t i = 0; i < CUDNN_DIM_MAX + 1; ++i)
        {
            std::cout << std::format("yDim[{}]: {}", i, yDim[i]) << std::endl;
        }

        if (yDim[2] == yDim[3])
        {
            //display(Y->hostPtr, yDim[2]);

            cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
            int64_t stride[4];
            Helpers::generateStrides(yDim, stride, 4, tensorFormat);
            for (int64_t h = 0; h < yDim[2]; ++h)
            {
                for (int64_t w = 0; w < yDim[3]; ++w)
                {
                    std::cout << std::setw(3) << std::setprecision(3) << Y->hostPtr[h * stride[2] + w * stride[3]] << " ";
                }
                std::cout << std::endl;
            }
        }
        else
        {
            display_flat(Y->hostPtr, yDim[2]);
        }
    }

    void Layer::printGrad()
    {
        if (!mGradSurface || !yTensor)
        {
            return;
        }

        cudaDeviceSynchronize();
        checkCudaErr(cudaMemcpy(mGradSurface->hostPtr, mGradSurface->devPtr, (size_t)(sizeof(mGradSurface->hostPtr[0]) * mGradSurface->n_elems), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        auto nbDims = yTensor->getDimCount();
        auto dims = yTensor->getDim();
        auto stride = yTensor->getStride();
        std::cout << std::format("{} dY:", mName.c_str()) << std::endl;

        for (size_t b = 0; b < dims[0]; ++b)
        {
            for (size_t h = 0; h < dims[nbDims - 2]; ++h)
            {
                for (int64_t w = 0; w < dims[nbDims - 1]; ++w)
                {
                    std::cout << mGradSurface->hostPtr[stride[0] * b + stride[nbDims - 2] * h + stride[nbDims - 1] * w] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    // Vars naming backfired once already. They really shouldn't match class members
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

        plan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));

        if (mVerbose) std::cout << "Plan tag: " << plan->getTag() << std::endl;

        workspace_size = plan->getWorkspaceSize();
        if (mVerbose) std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
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


    ConvBiasAct::ConvBiasAct(cudnnHandle_t& handle,
        const int64_t* inputDim,
        const int64_t kernelSize,
        const int64_t filterSize,
        cudnn_frontend::Tensor& inputTensor,
        void* inputDevPtr,
        const int64_t convPad,
        bool verbose)
        : Layer(handle, verbose)
    {
        // Defaults. TODO: Move to the appropriate place. Some of these are hyperparameters. Some are necessary constants. Bad place.
        constexpr int64_t alignment = 16; //16B to make Tensor cores work
        cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
        cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        constexpr int convDim = 2;
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        int64_t wTensorDim[] = { filterSize, inputDim[1], kernelSize, kernelSize }; // filter
        int64_t yTensorDim[] = { 0, 0, 0, 0 }; // Computed Below
        int64_t padA[] = { convPad, convPad };
        int64_t dilationA[] = { 1, 1 }; // TODO: make proper defaults
        int64_t convstrideA[] = { 1, 1 };
        int64_t bTensorDim[] = { 1, wTensorDim[0], 1, 1 };  // bias // should first parameter be equal to the batch size?? No? It's supposed to be shared across the batches, right?

        yTensorDim[0] = inputDim[0];
        yTensorDim[1] = wTensorDim[0];
        for (int dim = 0; dim < 2; dim++) {
            yTensorDim[dim + 2] = Helpers::getFwdConvOutputDim(inputDim[dim + 2], padA[dim], wTensorDim[dim + 2], convstrideA[dim], dilationA[dim]);
        }

        if (verbose)
        {
            std::cout << std::format("====DIMENSIONS====") << std::endl;
            std::cout << std::format("input dims are {}, {}, {}, {}", inputDim[0], inputDim[1], inputDim[2], inputDim[3]) << std::endl;
            std::cout << std::format("filter dims are {}, {}, {}, {}", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]) << std::endl;
            std::cout << std::format("output dims are {}, {}, {}, {}", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]) << std::endl;
        }

        int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

        W = std::make_unique<Helpers::Surface<float>>(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
        B = std::make_unique<Helpers::Surface<float>>(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
        Y = std::make_unique<Helpers::Surface<float>>(Ysize, false, 0.0f);

        try
        {
            //checkCudnnErr(cudnnCreate(&mHandle));

            // C1 - convolution
            int64_t stride[4];
            Helpers::generateStrides(wTensorDim, stride, 4, tensorFormat);
            auto wTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, wTensorDim)
                .setStride(4, stride)
                .setId('w')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            Helpers::generateStrides(bTensorDim, stride, 4, tensorFormat);
            auto bTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, bTensorDim)
                .setStride(4, stride)
                .setId('b')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            Helpers::generateStrides(yTensorDim, stride, 4, tensorFormat);
            auto afterConvTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, yTensorDim)
                .setStride(4, stride)
                .setId('A')  // after conv
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, yTensorDim)
                .setStride(4, stride)
                .setId('B')  // after bias
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            yTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
                .setDim(4, yTensorDim)
                .setStride(4, stride)
                .setId('y')  // after relu
                .setAlignment(alignment)
                .setDataType(dataType)
                .build());

            if (mVerbose)
            {
                std::cout << inputTensor.describe() << std::endl;
                std::cout << wTensor.describe() << std::endl;
                std::cout << afterConvTensor.describe() << std::endl;
                std::cout << yTensor->describe() << std::endl;
            }

            // Define the convolution problem
            auto convDesc = cudnn_frontend::ConvDescBuilder()
                .setComputeType(CUDNN_DATA_FLOAT)
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
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            if (mVerbose) std::cout << biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_RELU_FWD)//CUDNN_POINTWISE_SIGMOID_FWD
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            if (mVerbose) std::cout << actDesc.describe() << std::endl;

            // Create a convolution Node
            auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(inputTensor)
                .setwDesc(wTensor)
                .setyDesc(afterConvTensor) //afterConvTensor // yTensor
                .setcDesc(convDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            if (mVerbose) std::cout << conv_op.describe() << std::endl;

            // Create a Bias Node.
            auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(conv_op.getOutputTensor()) //conv_op.getOutputTensor()
                .setbDesc(bTensor)
                .setyDesc(afterBiasTensor) //afterBiasTensor
                .setpwDesc(biasDesc)
                .build();
            if (mVerbose) std::cout << bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(bias_op.getOutputTensor()) // bias_op.getOutputTensor()
                .setyDesc(*yTensor)
                .setpwDesc(actDesc)
                .build();
            if (mVerbose) std::cout << act_op.describe() << std::endl;

            // Create an Operation Graph.
            //std::vector<cudnn_frontend::Operation const*> ops = { &conv_op,  &bias_op, &act_op};
            std::vector<cudnn_frontend::Operation const*> ops;
            ops.emplace_back(&conv_op);
            ops.emplace_back(&bias_op);
            ops.emplace_back(&act_op);

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(mHandle)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            if (mVerbose) std::cout << opGraph.describe() << std::endl;

            plan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));

            if (mVerbose) std::cout << "Plan tag: " << plan->getTag() << std::endl;

            workspace_size = plan->getWorkspaceSize();
            if (mVerbose) std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }
            std::vector<void*> data_ptrs;
            data_ptrs.emplace_back(inputDevPtr);
            data_ptrs.emplace_back(W->devPtr); // Doesn't look so good. TODO: rethink naming
            data_ptrs.emplace_back(B->devPtr);
            data_ptrs.emplace_back(Y->devPtr);

            std::vector<int64_t> uids;
            uids.emplace_back(inputTensor.getId());
            uids.emplace_back('w');
            uids.emplace_back('b');
            uids.emplace_back('y');

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
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }

    Pool::Pool(cudnnHandle_t& handle, cudnn_frontend::Tensor& inputTensor, void* inputDevPtr, bool verfose)
        : Layer(handle, verfose)
    {
        int64_t poolTensorDim[] = { 0, 0, 0, 0 };
        auto inputDim = inputTensor.getDim();
        poolTensorDim[0] = inputDim[0];
        poolTensorDim[1] = inputDim[1];
        poolTensorDim[2] = inputDim[2] / 2;
        poolTensorDim[3] = inputDim[3] / 2;

        // TODO: Default parameters need to have a proper place
        int64_t windowDimPool[CUDNN_DIM_MAX] = { 2,2 };
        int64_t prePaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        int64_t postPaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        int64_t stridePool[CUDNN_DIM_MAX] = { 2,2 };

        if (mVerbose) std::cout << std::format("After pool dims are {}, {}, {}, {}", poolTensorDim[0], poolTensorDim[1], poolTensorDim[2], poolTensorDim[3]) << std::endl;


        int64_t Psize = poolTensorDim[0] * poolTensorDim[1] * poolTensorDim[2] * poolTensorDim[3];
        Y = std::make_unique<Helpers::Surface<float>>(Psize, false, 0.0f);

        try
        {
            // TODO: Defaults, place
            constexpr int64_t alignment = 16; //16
            cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
            cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
            float alpha = 1.0f;
            float beta = 0.0f;

            int64_t stride[4];

            auto const nanOpt = CUDNN_NOT_PROPAGATE_NAN;
            constexpr int64_t nbSpatialDims = 2;
            cudnn_frontend::cudnnResampleMode_t const mode = cudnn_frontend::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
            cudnn_frontend::cudnnPaddingMode_t const padding_mode = cudnn_frontend::cudnnPaddingMode_t::CUDNN_ZERO_PAD;

            Helpers::generateStrides(poolTensorDim, stride, 4, tensorFormat);

            yTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
                .setDim(4, poolTensorDim)
                .setStride(4, stride)
                .setId('p') //'p' // after pool // TODO: proper IDs
                .setAlignment(16)
                .setDataType(dataType)
                .build());
            if (mVerbose) std::cout << yTensor->describe() << std::endl;

            // Define the resample descriptor
            auto poolDesc = cudnn_frontend::ResampleDescBuilder()
                .setComputeType(CUDNN_DATA_FLOAT)
                .setNanPropagation(nanOpt)
                .setResampleMode(mode)
                .setPaddingMode(padding_mode)
                .setSpatialDim(nbSpatialDims, windowDimPool)
                .setSpatialStride(nbSpatialDims, stridePool)
                .setPrePadding(nbSpatialDims, prePaddingPool)
                .setPostPadding(nbSpatialDims, postPaddingPool)
                .build();
            if (mVerbose) std::cout << "Initialized Pool Desc" << std::endl;
            if (mVerbose) std::cout << poolDesc.describe() << std::endl;

            // Create a Resample Node
            auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                .setxDesc(inputTensor)
                .setyDesc(*yTensor)
                .setResampleDesc(poolDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            if (mVerbose) std::cout << pool_op.describe() << std::endl;

            std::vector<cudnn_frontend::Operation const*> ops = { &pool_op };

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(mHandle)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            if (mVerbose) std::cout << opGraph.describe() << std::endl;

            plan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));

            if (mVerbose) std::cout << "Plan tag: " << plan->getTag() << std::endl;

            workspace_size = plan->getWorkspaceSize();
            if (mVerbose) std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }

            workspace_size = plan->getWorkspaceSize();
            if (mVerbose) std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }

            std::vector<void*> data_ptrs;
            data_ptrs.emplace_back(inputDevPtr);
            data_ptrs.emplace_back(Y->devPtr);

            std::vector<int64_t> uids;
            uids.emplace_back(inputTensor.getId());
            uids.emplace_back(yTensor->getId());

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
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }

    //FC::FC(cudnnHandle_t& handle, cudnn_frontend::Tensor& inputTensor, void* inputDevPtr, int64_t numOutput, bool verbose)
    FC::FC(cudnnHandle_t& handle, Layer& prevLayer, int64_t numOutput, bool verbose, std::string name)
        : Layer(handle, verbose, name)
        , mPrevLayer(prevLayer)
    {
        cudnn_frontend::Tensor flatInputTensor = flattenTensor(mPrevLayer.getOutputTensor());
        auto inputDim = flatInputTensor.getDim();

        int64_t weightsTensorDim[] = { inputDim[0], inputDim[2], numOutput }; //batch K N
        int64_t outputTensorDim[] = { inputDim[0], inputDim[1], weightsTensorDim[2] }; //batch M N
        int64_t biasTensorDim[] = { 1, inputDim[1], numOutput };  //bias. It should be unique per output. And we need it in FC. Not sure about the output one though

        if (mVerbose)
        {
            printf("====DIMENSIONS====\n");
            std::cout << std::format("a matrix dims are {}, {}, {}", inputDim[0], inputDim[1], inputDim[2]) << std::endl;
            std::cout << std::format("b matrix dims are {}, {}, {}", weightsTensorDim[0], weightsTensorDim[1], weightsTensorDim[2]) << std::endl;
            std::cout << std::format("c matrix dims are {}, {}, {}", outputTensorDim[0], outputTensorDim[1], outputTensorDim[2]) << std::endl;
        }

        int64_t outputSize = outputTensorDim[0] * outputTensorDim[1] * outputTensorDim[2];

        W = std::make_unique<Helpers::Surface<float>>(weightsTensorDim[0] * weightsTensorDim[1] * weightsTensorDim[2]);
        B = std::make_unique<Helpers::Surface<float>>(biasTensorDim[0] * biasTensorDim[1] * biasTensorDim[2]); // bias TODO: doublecheck
        Z = std::make_unique<Helpers::Surface<float>>(outputSize); // after bias
        Y = std::make_unique<Helpers::Surface<float>>(outputSize, false, 0.0f);

        constexpr int64_t nbDims = 3;
        int64_t FCstride[nbDims];

        try
        {
            // TODO: Defaults, place

            //constexpr int64_t alignment = 16; //16
            //cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
            //cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
            //float alpha = 1.0f;
            //float beta = 0.0f;
                // Creates the necessary tensor descriptors

            //Helpers::generateStrides(inputDim, FCstride, nbDims, tensorFormat); // are strides correct? this function might be off for 3d tensor
            Helpers::generateStrides(weightsTensorDim, FCstride, nbDims, tensorFormat);
            auto bMatrixTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, weightsTensorDim)
                .setStride(nbDims, FCstride)
                .setId(getTensorId()) //'b'
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            Helpers::generateStrides(biasTensorDim, FCstride, nbDims, tensorFormat);
            auto biasTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, biasTensorDim)
                .setStride(nbDims, FCstride)
                .setId(getTensorId()) // 'z'
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            Helpers::generateStrides(outputTensorDim, FCstride, nbDims, tensorFormat);
            auto afterMatMulTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, outputTensorDim)
                .setStride(nbDims, FCstride)
                .setId(getTensorId())  // 'A' after matmul
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();
            auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, outputTensorDim)
                .setStride(nbDims, FCstride)
                .setId(getTensorId())  // 'B' after bias
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();
            yTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
                .setDim(nbDims, outputTensorDim)
                .setStride(nbDims, FCstride)
                .setId(getTensorId())  // 'c' output after gelu
                .setAlignment(alignment)
                .setDataType(dataType)
                .build());

            if (mVerbose)
            {
                std::cout << flatInputTensor.describe() << std::endl;
                std::cout << bMatrixTensor.describe() << std::endl;
                std::cout << biasTensor.describe() << std::endl;
                std::cout << afterMatMulTensor.describe() << std::endl;
                std::cout << afterBiasTensor.describe() << std::endl;
                std::cout << yTensor->describe() << std::endl;
            }

            // Define the bias descriptor
            auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            if (mVerbose) std::cout << biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto actDesc = cudnn_frontend::PointWiseDescBuilder()
#if (CUDNN_VERSION >= 8500)
                .setMode(CUDNN_POINTWISE_GELU_APPROX_TANH_FWD)
#else
                .setMode(CUDNN_POINTWISE_GELU_FWD)
#endif
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            if (mVerbose) std::cout << actDesc.describe() << std::endl;

            // Define the matmul desc
            auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
            if (mVerbose) std::cout << matmulDesc.describe() << std::endl;

            // Create a matmul Node
            auto matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                .setaMatDesc(flatInputTensor)
                .setbMatDesc(bMatrixTensor)
                .setcMatDesc(afterMatMulTensor)
                .setmatmulDesc(matmulDesc)
                .build();
            if (mVerbose) std::cout << matmul_op.describe() << std::endl;

            // Create a Bias Node.
            auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(matmul_op.getOutputTensor())
                .setbDesc(biasTensor)
                .setyDesc(afterBiasTensor)
                .setpwDesc(biasDesc)
                .build();
            if (mVerbose) std::cout << bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(bias_op.getOutputTensor())
                .setyDesc(*yTensor)
                .setpwDesc(actDesc)
                .build();
            if (mVerbose) std::cout << act_op.describe() << std::endl;

            // Create an Operation Graph. In this case it is matmul bias activation
            std::array<cudnn_frontend::Operation const*, 3> ops = { &matmul_op, &bias_op, &act_op };

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(mHandle)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            if (mVerbose) std::cout << opGraph.describe() << std::endl;

            plan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));

            if (mVerbose) std::cout << "Plan tag: " << plan->getTag() << std::endl;

            workspace_size = plan->getWorkspaceSize();
            if (mVerbose) std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }

            std::vector<void*> data_ptrs;
            data_ptrs.emplace_back(mPrevLayer.getOutputSurface().devPtr);
            data_ptrs.emplace_back(W->devPtr);
            data_ptrs.emplace_back(Y->devPtr);
            data_ptrs.emplace_back(Z->devPtr);
            data_ptrs.emplace_back(B->devPtr);

            std::vector<int64_t> uids;
            uids.emplace_back(flatInputTensor.getId());
            uids.emplace_back(bMatrixTensor.getId());
            uids.emplace_back(yTensor->getId());
            uids.emplace_back(biasTensor.getId());
            uids.emplace_back(afterBiasTensor.getId());

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
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }

        mGradSurface = std::make_unique<Helpers::Surface<float>>(outputTensorDim[0] * outputTensorDim[1] * outputTensorDim[2], false, 0.0f);
        Helpers::generateStrides(outputTensorDim, FCstride, nbDims, tensorFormat);
        mGradTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
            .setDim(nbDims, outputTensorDim)
            .setStride(nbDims, FCstride)
            .setId(getTensorId())
            .setAlignment(alignment)
            .setDataType(dataType)
            .build());

        _setupGradient(biasTensorDim, weightsTensorDim, inputDim);
    }

    void FC::backpropagate()
    {
        auto checkStatus = [](cudnnStatus_t status)
        {
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << cudnnGetErrorString(status) << std::endl;
                return; // something went wrong during execution. TODO: return status instead???
            }
        };
        std::cout << std::format("{} backprop", mName.c_str()) << std::endl;

        try
        {
            //calculate gradient
            if (/*!mBiasGradPlan || !mBiasGradVariantPack ||*/ !mWeightsGradPlan || !mWeightsGradVariantPack || !mDataGradPlan || !mDataGradVariantPack)
            {
                return; // not initialized
            }

            //checkStatus(cudnnBackendExecute(mHandle, mBiasGradPlan->get_raw_desc(),    mBiasGradVariantPack->get_raw_desc()));
            checkStatus(cudnnBackendExecute(mHandle, mWeightsGradPlan->get_raw_desc(), mWeightsGradVariantPack->get_raw_desc()));
            checkStatus(cudnnBackendExecute(mHandle, mDataGradPlan->get_raw_desc(), mDataGradVariantPack->get_raw_desc()));

            //std::cout << "Creating reduce tensor" << std::endl;
            cudnnReduceTensorDescriptor_t reduceTensorDesc;
            checkStatus(cudnnCreateReduceTensorDescriptor(&reduceTensorDesc));

            //std::cout << "Setting reduce tensor" << std::endl;
            checkStatus(cudnnSetReduceTensorDescriptor(
                /*cudnnReduceTensorDescriptor_t   */reduceTensorDesc,
                /*cudnnReduceTensorOp_t           */CUDNN_REDUCE_TENSOR_ADD,
                /*cudnnDataType_t                 */dataType,
                /*cudnnNanPropagation_t           */CUDNN_NOT_PROPAGATE_NAN,
                /*cudnnReduceTensorIndices_t      */CUDNN_REDUCE_TENSOR_NO_INDICES,
                /*cudnnIndicesType_t              */CUDNN_64BIT_INDICES));

            //std::cout << "Creating grad tensors" << std::endl;
            cudnnTensorDescriptor_t mGradTensorDesc, mBiasGradTensorDesc;
            checkStatus(cudnnCreateTensorDescriptor(&mGradTensorDesc));
            checkStatus(cudnnCreateTensorDescriptor(&mBiasGradTensorDesc));

            // TODO: Proper defaults
            constexpr int nbDims = 3;

            //auto& inputTensor = prevLayer.getOutputTensor();

            //assert(nbDims == inputTensor.getDimCount());

            auto gradDim = yTensor->getDim();
            int mGradDims[] = { static_cast<int>(gradDim[0]), static_cast<int>(gradDim[1]), static_cast<int>(gradDim[2]) };
            int gradStride[] = { gradDim[1] * gradDim[2] , gradDim[2], 1 };

            int mBiasGradDims[] = { 1, gradDim[1], gradDim[2] }; // reduced over batch
            //int mBiasGradStride[] = { gradDim[1] * gradDim[2] , gradDim[2], 1 };

            //cudnnReduc

            //std::cout << "Setting grad tensors" << std::endl;
            checkStatus(cudnnSetTensorNdDescriptor(mGradTensorDesc,
                dataType,
                nbDims,
                mGradDims,
                gradStride));
            checkStatus(cudnnSetTensorNdDescriptor(mBiasGradTensorDesc,
                dataType,
                nbDims,
                mBiasGradDims,
                gradStride));

            //std::cout << "Getting workspace size" << std::endl;
            // Get workspace size for reduce operation
            size_t mBiasGradWorkspaceSize;
            checkStatus(cudnnGetReductionWorkspaceSize(mHandle, reduceTensorDesc, mGradTensorDesc, mBiasGradTensorDesc, &mBiasGradWorkspaceSize));
            //std::cout << std::format("Reduction op will need {} size workspace", mBiasGradWorkspaceSize) << std::endl;

            // Allocate memory for workspace
            void* workspace;
            cudaMalloc(&workspace, mBiasGradWorkspaceSize);

            //std::cout << "Reducing tensor" << std::endl;
            checkStatus(cudnnReduceTensor(
                /*cudnnHandle_t */                          mHandle,
                /*const cudnnReduceTensorDescriptor_t*/     reduceTensorDesc,
                /*void**/ nullptr,
                /*size_t*/                                  0,
                /*void**/ workspace,
                /*size_t*/                                  mBiasGradWorkspaceSize,
                /*const void**/ &alpha,
                /*const cudnnTensorDescriptor_t*/           mGradTensorDesc,
                /*const void**/ mGradSurface->devPtr,
                /*const void**/ &beta,
                /*const cudnnTensorDescriptor_t*/           mBiasGradTensorDesc,
                /*void**/ mBiasGradSurface->devPtr));

            // SGD
            // update bias
            cublasSaxpy(mBiasGradSurface->n_elems,
                -epsilon, // learning rate
                mBiasGradSurface->devPtr, 1,
                B->devPtr, 1);

            // update weights
            cublasSaxpy(mWeightsGradSurface->n_elems,
                -epsilon, // learning rate
                mWeightsGradSurface->devPtr, 1,
                W->devPtr, 1);
        }
        catch (std::exception& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }

    void FC::printBias()
    {
        std::cout << "FC bias debug:" << std::endl;

        //cudaDeviceSynchronize();
        //checkCudaErr(cudaMemcpy(B->hostPtr, B->devPtr, sizeof(B->hostPtr[0]) * B->n_elems, cudaMemcpyDeviceToHost));
        //cudaDeviceSynchronize();

        B->devToHostSync();

        for (int64_t i = 0; i < B->n_elems; ++i)
        {
            std::cout << std::setprecision(3) << std::format("{} ", B->hostPtr[i]);
        }

        std::cout << std::endl;

        if (mBiasGradSurface)
        {
            std::cout << "FC bias grad debug:" << std::endl;

            mBiasGradSurface->devToHostSync();

            for (int64_t i = 0; i < mBiasGradSurface->n_elems; ++i)
            {
                std::cout << std::setprecision(3) << std::format("{} ", mBiasGradSurface->hostPtr[i]);
            }

            std::cout << std::endl;
        }

    }

    void FC::printWeights()
    {
        std::cout << std::format("{} weights debug:", mName.c_str()) << std::endl;

        if (W)
        {
            W->devToHostSync();

            for (int64_t i = 0; i < W->n_elems; ++i)
            {
                std::cout << std::setprecision(3) << std::format("{} ", W->hostPtr[i]);
            }

            std::cout << std::endl;
        }

        if (mWeightsGradSurface)
        {
            std::cout << std::format("{} weights grad debug", mName.c_str()) << std::endl;

            mWeightsGradSurface->devToHostSync();

            for (int64_t i = 0; i < mWeightsGradSurface->n_elems; ++i)
            {
                std::cout << std::setprecision(3) << std::format("{} ", mWeightsGradSurface->hostPtr[i]);
            }

            std::cout << std::endl;
        }
    }

    cudnn_frontend::Tensor FC::flattenTensor(cudnn_frontend::Tensor& tensor)
    {

        // TODO: magic numbers remove I shall
        if (tensor.getDimCount() > 3)
        {
            auto tensorDim = tensor.getDim();
            int64_t flattenTensorDim[] = { tensorDim[0], 1, tensorDim[1] * tensorDim[2] * tensorDim[3] }; // can be done better

            // TODO: Defaults, place
            constexpr int64_t alignment = 16; //16
            cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
            cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
            float alpha = 1.0f;
            float beta = 0.0f;
            int64_t FCstride[3];
            Helpers::generateStrides(flattenTensorDim, FCstride, 3, tensorFormat);
            // RVO
            return cudnn_frontend::TensorBuilder()
                .setDim(3, flattenTensorDim)
                .setStride(3, FCstride)
                .setId(getTensorId())
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
                .setId(getTensorId())
                .setAlignment(tensor.getAlignment())
                .setDataType(static_cast<cudnnDataType_t>(tensor.getDataType()))
                .build();
        }
    }

    void FC::_setupGradient(int64_t biasDim[], int64_t weightsTensorDim[], const int64_t inputDim[])
    {
        //auto outputDim = yTensor->getDim();
        //mGradSurface = std::make_unique<Helpers::Surface<float>>(outputDim[0] * outputDim[1] * outputDim[2], false, 0.0f);

        try
        {
            _setupBiasGrad(biasDim);
            _setupWeightsGrad(weightsTensorDim, inputDim);
            //_setupDataGrad();
        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }

    void FC::_setupBiasGrad(int64_t biasDim[])
    {
        if (mVerbose) std::cout << "_setupBiasGrad" << std::endl;
        auto outputDim = yTensor->getDim();
        auto nbDims = yTensor->getDimCount();
        mBiasGradSurface = std::make_unique<Helpers::Surface<float>>(biasDim[0] * biasDim[1] * biasDim[2], false, 0.5f);
        if (mVerbose) std::cout << mGradTensor->describe() << std::endl;

        int64_t stride[3] = { 10, 1, 1 };
        //Helpers::generateStrides(biasDim, stride, nbDims, CUDNN_TENSOR_NCHW);
        auto biasGradTensor = cudnn_frontend::TensorBuilder()
            .setAlignment(alignment)
            .setDataType(dataType)
            .setDim(nbDims, biasDim)
            .setStride(nbDims, stride)
            .setId(getTensorId())
            .build();
        if (mVerbose) std::cout << biasGradTensor.describe() << std::endl;

        // Define the reduction descriptor
        auto reductionDesc = cudnn_frontend::ReductionDescBuilder()
            .setComputeType(dataType)
            .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
            .build();
        if (mVerbose) std::cout << reductionDesc.describe() << std::endl;

        // Create a reduction add Node.
        auto reduction_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
            .setxDesc(*mGradTensor)
            .setyDesc(biasGradTensor)
            .setreductionDesc(reductionDesc)
            .build();
        if (mVerbose) std::cout << reduction_op.describe() << std::endl;

        //if (mVerbose) std::cout << "avgReductionDesc" << std::endl;
        //auto avgReductionDesc = ReductionDescBuilder()
        //    .setComputeType(dataType)
        //    .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
        //    .build();
        //if (mVerbose) std::cout << avgReductionDesc.describe() << std::endl;

        //if (mVerbose) std::cout << "avgReduction_op" << std::endl;
        //auto avgReduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
        //    .setxDesc(pwLog_op.getOutputTensor())
        //    .setyDesc(lossTensor)
        //    .setreductionDesc(avgReductionDesc)
        //    .build();
        //if (mVerbose) std::cout << avgReduction_op.describe() << std::endl;

        //auto pw_add_desc = cudnn_frontend::PointWiseDescBuilder()
        //    .setComputeType(dataType)
        //    .setMode(CUDNN_POINTWISE_ADD)
        //    .build();

        //auto pw_add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_POINTWISE_DESCRIPTOR)
        //    .setxDesc(*mGradTensor)
        //    .setyDesc(biasGradTensor)
        //    .setpwDesc(pw_add_desc)
        //    .build();

        std::vector<cudnn_frontend::Operation const*> ops = { &reduction_op };
        std::vector<void*> data_ptrs = { mGradSurface->devPtr, mBiasGradSurface->devPtr };
        std::vector<int64_t> uids = { mGradTensor->getId(), biasGradTensor.getId() };

        //_setPlan(ops, data_ptrs, uids, mBiasGradPlan, mBiasGradVariantPack, mBiasGradWorkspaceSize, mBiasGradWorkspacePtr);

        //if (mVerbose) std::cout << "_setPlan Raw" << std::endl;

        //auto opGraph = cudnn_frontend::OperationGraphBuilder()
        //    .setHandle(mHandle)
        //    .setOperationGraph(ops.size(), ops.data())
        //    .build();

        //if (mVerbose) std::cout << opGraph.describe() << std::endl;

        //mBiasGradPlan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));

        //if (mVerbose) std::cout << "Plan tag: " << mBiasGradPlan->getTag() << std::endl;

        //mBiasGradWorkspaceSize = mBiasGradPlan->getWorkspaceSize();
        //if (mVerbose) std::cout << mBiasGradPlan->describe() << " requires workspace " << mBiasGradWorkspaceSize << std::endl;

        //if (mBiasGradWorkspaceSize > 0) {
        //    checkCudaErr(cudaMalloc(&mBiasGradWorkspacePtr, (size_t)mBiasGradWorkspaceSize));
        //}

        //assert(data_ptrs.size() == uids.size());
        //int64_t num_ptrs = data_ptrs.size();
        //if (mVerbose) std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
        //mBiasGradVariantPack = std::make_unique<cudnn_frontend::VariantPack>(cudnn_frontend::VariantPackBuilder()
        //    .setWorkspacePointer(mBiasGradWorkspacePtr)
        //    .setDataPointers(num_ptrs, data_ptrs.data())
        //    .setUids(num_ptrs, uids.data())
        //    .build());
        //if (mVerbose) std::cout << "mBiasGradVariantPack " << mBiasGradVariantPack->describe() << std::endl;
    }

    void FC::_setupWeightsGrad(int64_t weightsTensorDim[], const int64_t inputDim[])
    {
        auto nbDims = yTensor->getDimCount();
        //int64_t stride[3];
        // this is a vector, so this shall be a valid transpose
        int64_t transposedInputDim[] = { inputDim[0], inputDim[2], inputDim[1] };
        int64_t transposedInputStride[] = { transposedInputDim[1] * transposedInputDim[2], 1, 1 };

        mWeightsGradSurface = std::make_unique<Helpers::Surface<float>>(W->n_elems, false, 0.5f);

        int64_t weightsStride[] = { weightsTensorDim[1] * weightsTensorDim[2], weightsTensorDim[2], 1 };
        if (mVerbose) std::cout << mGradTensor->describe() << std::endl;

        auto transposedIntputTensor = cudnn_frontend::TensorBuilder()
            .setAlignment(alignment)
            .setDataType(dataType)
            .setDim(nbDims, transposedInputDim)
            .setStride(nbDims, transposedInputStride)
            .setId(getTensorId())
            .build();
        if (mVerbose) std::cout << transposedIntputTensor.describe() << std::endl;

        //Helpers::generateStrides(weightsTensorDim, stride, nbDims, tensorFormat);
        auto weightsGradTensor = cudnn_frontend::TensorBuilder()
            .setAlignment(alignment)
            .setDataType(dataType)
            .setDim(nbDims, weightsTensorDim)
            .setStride(nbDims, weightsStride)
            .setId(getTensorId())
            .build();
        if (mVerbose) std::cout << weightsGradTensor.describe() << std::endl;

        // Define the matmul desc
        auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
        if (mVerbose) std::cout << matmulDesc.describe() << std::endl;

        // Create a matmul Node
        auto matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
            .setaMatDesc(transposedIntputTensor)
            .setbMatDesc(*mGradTensor)
            .setcMatDesc(weightsGradTensor)
            .setmatmulDesc(matmulDesc)
            .build();
        if (mVerbose) std::cout << matmul_op.describe() << std::endl;

        std::vector<cudnn_frontend::Operation const*> ops = { &matmul_op };
        std::vector<void*> data_ptrs = { mPrevLayer.getOutputSurface().devPtr, mGradSurface->devPtr, mWeightsGradSurface->devPtr };
        std::vector<int64_t> uids = { transposedIntputTensor.getId(), mGradTensor->getId(), weightsGradTensor.getId() };

        _setPlan(ops, data_ptrs, uids, mWeightsGradPlan, mWeightsGradVariantPack, mWeightsGradWorkspaceSize, mWeightsGradWorkspacePtr);
    }

    void FC::_setupDataGrad(int64_t weightsTensorDim[], const int64_t inputDim[])
    {
        auto nbDims = yTensor->getDimCount();
        int64_t inputStride[] = { inputDim[1] * inputDim[2], 1, 1 }; // should be flatten

        int64_t weightsStride[] = { weightsTensorDim[1] * weightsTensorDim[2], weightsTensorDim[2], 1 };
        if (mVerbose) std::cout << mGradTensor->describe() << std::endl;


        // TODO: deduplicate. this tensore is already setup in forward propagation
        auto intputTensor = cudnn_frontend::TensorBuilder()
            .setAlignment(alignment)
            .setDataType(dataType)
            .setDim(nbDims, inputDim)
            .setStride(nbDims, inputStride)
            .setId(getTensorId())
            .build();
        if (mVerbose) std::cout << intputTensor.describe() << std::endl;

        //Helpers::generateStrides(weightsTensorDim, stride, nbDims, tensorFormat);
        auto weightsGradTensor = cudnn_frontend::TensorBuilder()
            .setAlignment(alignment)
            .setDataType(dataType)
            .setDim(nbDims, weightsTensorDim)
            .setStride(nbDims, weightsStride)
            .setId(getTensorId())
            .build();
        if (mVerbose) std::cout << weightsGradTensor.describe() << std::endl;

        // Define the matmul desc
        auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
        if (mVerbose) std::cout << matmulDesc.describe() << std::endl;

        // Create a matmul Node
        auto matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
            .setaMatDesc(intputTensor)
            .setbMatDesc(*mGradTensor)
            .setcMatDesc(weightsGradTensor)
            .setmatmulDesc(matmulDesc)
            .build();
        if (mVerbose) std::cout << matmul_op.describe() << std::endl;

        std::vector<cudnn_frontend::Operation const*> ops = { &matmul_op };
        std::vector<void*> data_ptrs = { mPrevLayer.getOutputSurface().devPtr, mGradSurface->devPtr, mWeightsGradSurface->devPtr };
        std::vector<int64_t> uids = { intputTensor.getId(), mGradTensor->getId(), weightsGradTensor.getId() };

        _setPlan(ops, data_ptrs, uids, mWeightsGradPlan, mWeightsGradVariantPack, mWeightsGradWorkspaceSize, mWeightsGradWorkspacePtr);
    }

    //Softmax::Softmax(cudnnHandle_t& handle, 
    //                 cudnn_frontend::Tensor& inputTensor, 
    //                 Helpers::Surface<float>& srcSurface,
    //                 //void* inputDevPtr, 
    //                 bool verbose)
    //    : Layer(handle, verbose)
    //    , mSrcSurface(srcSurface)
    //{
    //    // We expect to have a flatten input
    //    // Since we use backend here - we won't set a frontend descriptor
    //    //cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    //
    //    cudnnCreateTensorDescriptor(&srcTensorDesc);
    //    cudnnCreateTensorDescriptor(&sftTensorDesc);
    //
    //    // TODO: Proper defaults
    //    constexpr int nbDims = 3;
    //
    //    assert(nbDims == inputTensor.getDimCount());
    //    
    //    auto inputDim = inputTensor.getDim();
    //    // Using library notation
    //    //int dimA[] = { inputDim[0], inputDim[1], inputDim[2] };
    //    // We don't expect to have more than INT_MAX elements in dimensions, so this should be safe... Rigth?
    //    mDims = { static_cast<int>(inputDim[0]), static_cast<int>(inputDim[1]), static_cast<int>(inputDim[2]) };
    //    int strideA[] = { inputDim[1] * inputDim[2] , inputDim[2], 1};
    //
    //    Y = std::make_unique<Helpers::Surface<float>>(mDims[0] * mDims[1] * mDims[2], false, 0.0f);
    //    
    //    cudnnSetTensorNdDescriptor(srcTensorDesc,
    //        CUDNN_DATA_FLOAT,
    //        nbDims,
    //        mDims.data(),
    //        strideA);
    //    cudnnSetTensorNdDescriptor(sftTensorDesc,
    //        CUDNN_DATA_FLOAT,
    //        nbDims,
    //        mDims.data(),
    //        strideA);
    //
    //    // we need to define output
    //    yTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
    //        .setDim(3, inputDim)
    //        .setStride(3, inputTensor.getStride())
    //        .setId(getTensorId())
    //        .setAlignment(16)
    //        .setDataType(CUDNN_DATA_FLOAT)
    //        .build());
    //}

    Softmax::Softmax(cudnnHandle_t& handle, Layer& prevLayer, bool verbose, std::string name)
        : Layer(handle, verbose, std::move(name))
        //, mSrcSurface(prevLayer.getOutputSurface())
        , mPrevLayer(prevLayer)
    {
        // We expect to have a flatten input
        // Since we use backend here - we won't set a frontend descriptor
        //cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;

        cudnnCreateTensorDescriptor(&srcTensorDesc);
        cudnnCreateTensorDescriptor(&sftTensorDesc);

        // TODO: Proper defaults
        constexpr int nbDims = 3;

        auto& inputTensor = prevLayer.getOutputTensor();

        assert(nbDims == inputTensor.getDimCount());

        auto inputDim = inputTensor.getDim();
        // We don't expect to have more than INT_MAX elements in dimensions, so this should be safe... Rigth?
        mDims = { static_cast<int>(inputDim[0]), static_cast<int>(inputDim[1]), static_cast<int>(inputDim[2]) };
        int stride[] = { inputDim[1] * inputDim[2] , inputDim[2], 1 };

        Y = std::make_unique<Helpers::Surface<float>>(mDims[0] * mDims[1] * mDims[2], false, 0.0f);

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
        yTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
            .setDim(3, inputDim)
            .setStride(3, inputTensor.getStride())
            .setId(getTensorId())
            .setAlignment(16)
            .setDataType(CUDNN_DATA_FLOAT)
            .build());

        // gradient surface has same dimensions as the output
        mGradSurface = std::make_unique<Helpers::Surface<float>>(mDims[0] * mDims[1] * mDims[2], false, 0.0f);
    }

    int Softmax::executeInference()
    {
        // TODO: Proper defaults
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;

        auto status = cudnnSoftmaxForward(mHandle,
            CUDNN_SOFTMAX_ACCURATE,
            CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha,
            srcTensorDesc,
            //mSrcSurface.devPtr,
            mPrevLayer.getOutputSurface().devPtr,
            &beta,
            sftTensorDesc,
            Y->devPtr);

        if (mVerbose)
        {
            std::cout << cudnnGetErrorString(status) << std::endl;
        }

        return 0; // Ok
    }

    void Softmax::printOutput()
    {
        //TODO: deduplicate

        if (!Y)
        {
            throw;
        }

        cudaDeviceSynchronize();
        checkCudaErr(cudaMemcpy(Y->hostPtr, Y->devPtr, (size_t)(sizeof(Y->hostPtr[0]) * Y->n_elems), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

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
                std::cout << std::setprecision(3) << mPrevLayer.getOutputSurface().hostPtr[batchStride * i + j] << " ";
            }

            std::cout << std::format("\nY{}", i) << std::endl;
            for (size_t j = 0; j < batchStride; ++j)
            {
                std::cout << std::setprecision(3) << Y->hostPtr[batchStride * i + j] << " ";
            }
            std::cout << std::endl;

        }
    }

    void Softmax::backpropagate()
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
                Y->devPtr,
                /*dyDesc*/ srcTensorDesc, // srcTensorDesc == sftTensorDesc, probably can be used interchangeably
                /**dy*/ mGradSurface->devPtr, // we expect valid gradient to be populated. make a sanity check?
                &beta,
                srcTensorDesc, // that makes sense, right? it describes the data pointer in the other layer
                mPrevLayer.getGradSurface().devPtr); // next level in backprop chain needs the gradient

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



    //static cudnn_frontend::Tensor tensor_create(cudnnDataType_t type,
    //    int64_t id,
    //    int64_t const* dim,
    //    int64_t const* stride,
    //    bool is_virtual,
    //    bool is_value) {
    //    int nbDims = 3;
    //    auto tensor_created = cudnn_frontend::TensorBuilder()
    //        .setDim(nbDims, dim)
    //        .setStride(nbDims, stride)
    //        .setId(id)
    //        .setAlignment(16) // 16B alignment is needed to run a tensor core engine
    //        .setDataType(type)
    //        .setVirtual(is_virtual)
    //        .setByValue(is_value)
    //        .build();
    //    std::cout << tensor_created.describe() << std::endl;
    //    return tensor_created;
    //};
    //
    //static cudnn_frontend::PointWiseDesc pw_desc_create(cudnnDataType_t type, cudnnPointwiseMode_t mode) {
    //    auto pw_desc_created = cudnn_frontend::PointWiseDescBuilder()
    //        .setMode(mode)
    //        .setComputeType(type)
    //        .build();
    //
    //    std::cout << pw_desc_created.describe() << std::endl;
    //    return pw_desc_created;
    //}
    //
    //static cudnn_frontend::Operation unary_pw_op_create(cudnn_frontend::Tensor const& xDesc, cudnn_frontend::Tensor const& yDesc,
    //    cudnn_frontend::PointWiseDesc const& pwDesc) {
    //    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    //        .setxDesc(xDesc)
    //        .setyDesc(yDesc)
    //        .setpwDesc(pwDesc)
    //        .build();
    //    std::cout << pw_op_created.describe() << std::endl;
    //    return pw_op_created;
    //}
    //
    //static cudnn_frontend::Operation binary_pw_op_create(cudnn_frontend::Tensor const& xDesc, cudnn_frontend::Tensor const& bDesc,
    //    cudnn_frontend::Tensor const& yDesc, cudnn_frontend::PointWiseDesc const& pwDesc) {
    //    auto pw_op_created = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    //        .setxDesc(xDesc)
    //        .setbDesc(bDesc)
    //        .setyDesc(yDesc)
    //        .setpwDesc(pwDesc)
    //        .build();
    //    std::cout << pw_op_created.describe() << std::endl;
    //    return pw_op_created;
    //}
    //
    //
    //static cudnn_frontend::Tensor createSoftmaxForward(int64_t b,
    //    int64_t h,
    //    int64_t s_q,
    //    //int64_t s_kv,
    //    //int64_t d,
    //    //MHA_Layout layout,
    //    //bool enable_dropout,
    //    bool softmax_output_virtual,
    //    cudnnDataType_t tensorType,
    //    std::vector<cudnn_frontend::Operation>& ops,
    //    cudnn_frontend::Tensor& prevBlockOutputTensor) {
    //    //CUDNN_FRONTEND_UNUSED(d);
    //    //CUDNN_FRONTEND_UNUSED(layout);
    //
    //
    //    //int64_t afterBMM1_dim[4] = { b, h, s_q, s_kv };
    //    //int64_t afterBMM1_stride[4] = { h * s_q * s_kv, s_q * s_kv, s_kv, 1 };
    //
    //    //int64_t afterReduction_dim[4] = { b, h, s_q, 1 };
    //    //int64_t afterReduction_stride[4] = { h * s_q, s_q, 1, 1 };
    //
    //    int nbDims = 3;
    //
    //    int64_t afterBMM1_dim[] = { b, h, s_q };
    //    int64_t afterBMM1_stride[] = { h * s_q, s_q, 1 };
    //
    //    int64_t afterReduction_dim[] = { b, h, 1 };
    //    int64_t afterReduction_stride[] = { h, 1, 1 };
    //
    //    std::cout << "Creating Softmax Forward" << std::endl;
    //    std::cout << std::format("afterBMM1_dim {} {} {}", b, h, s_q) << std::endl;
    //    std::cout << std::format("afterReduction_dim {} {} {}", b, h, 1) << std::endl;
    //
    //
    //    cudnnDataType_t softmaxOutputType = tensorType; //(enable_dropout || softmax_output_virtual) ? CUDNN_DATA_FLOAT : tensorType;
    //    //uint64_t softmaxOutputName = softmax_output_virtual ? VIRTUAL_ID + 154 : S_ID;
    //    uint64_t softmaxOutputName = Layer::getTensorId();
    //    // max (x)
    //    std::cout << "afterMaxReductionTensor" << std::endl;
    //    auto afterMaxReductionTensor = tensor_create(CUDNN_DATA_FLOAT, Layer::getTensorId(), afterReduction_dim, afterReduction_stride, true, false); // is virtual
    //    // x - max(x)
    //    std::cout << "afterSubtractionTensor" << std::endl;
    //    auto afterSubtractionTensor = tensor_create(CUDNN_DATA_FLOAT, Layer::getTensorId(), afterBMM1_dim, afterBMM1_stride, true, false); // is virtual
    //    // e^(x - max(x))
    //    std::cout << "afterExponentTensor" << std::endl;
    //    auto afterExponentTensor = tensor_create(CUDNN_DATA_FLOAT, Layer::getTensorId(), afterBMM1_dim, afterBMM1_stride, true, false); // is virtual;
    //    // sum (e^(x - max(x)))
    //    std::cout << "afterAddReductionTensor" << std::endl;
    //    auto afterAddReductionTensor = tensor_create(CUDNN_DATA_FLOAT, Layer::getTensorId(), afterReduction_dim, afterReduction_stride, true, false); // is virtual
    //
    //    // divide (e/ sum(e))
    //    std::cout << "afterDivisionTensor" << std::endl;
    //    auto afterDivisionTensor = cudnn_frontend::TensorBuilder()
    //        .setDim(nbDims, afterBMM1_dim)
    //        .setStride(nbDims, afterBMM1_stride)
    //        .setId(softmaxOutputName)
    //        .setAlignment(16) // 16B alignment is needed to run a tensor core engine
    //        .setDataType(softmaxOutputType)
    //        .setVirtual(softmax_output_virtual)
    //        .setByValue(false)
    //        //.setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16) // do I need it?
    //        .build();
    //    std::cout << afterDivisionTensor.describe() << std::endl;
    //
    //    // Define the reduction descriptor
    //    std::cout << "reductionMaxDesc" << std::endl;
    //    auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
    //        .setComputeType(CUDNN_DATA_FLOAT)
    //        .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
    //        .build();
    //    std::cout << reductionMaxDesc.describe() << std::endl;
    //
    //    // Create a reduction max Node.
    //    std::cout << "reductionMax_op" << std::endl;
    //    auto reductionMax_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
    //        .setxDesc(prevBlockOutputTensor)
    //        .setyDesc(afterMaxReductionTensor)
    //        .setreductionDesc(reductionMaxDesc)
    //        .build();
    //    std::cout << reductionMax_op.describe() << std::endl;
    //
    //    // Define the subtract descriptor
    //    std::cout << "subtractDesc" << std::endl;
    //    auto subtractDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_SUB);
    //
    //    // Create a subtract Node.
    //    std::cout << "subtract_op" << std::endl;
    //    auto subtract_op = binary_pw_op_create(prevBlockOutputTensor, afterMaxReductionTensor, afterSubtractionTensor, subtractDesc);
    //
    //    // Define the exponent descriptor
    //    std::cout << "exponentDesc" << std::endl;
    //    auto exponentDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_EXP);
    //
    //    // Create a exponent Node.
    //    std::cout << "exponent_op" << std::endl;
    //    auto exponent_op = unary_pw_op_create(afterSubtractionTensor, afterExponentTensor, exponentDesc);
    //
    //    // Define the reduction descriptor
    //    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
    //        .setComputeType(CUDNN_DATA_FLOAT)
    //        .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
    //        .build();
    //    std::cout << reductionAddDesc.describe() << std::endl;
    //
    //    // Create a reduction add Node.
    //    auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
    //        .setxDesc(afterExponentTensor)
    //        .setyDesc(afterAddReductionTensor)
    //        .setreductionDesc(reductionAddDesc)
    //        .build();
    //
    //    std::cout << reductionAdd_op.describe() << std::endl;
    //
    //    // Define the division descriptor
    //    auto divisionDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_DIV);
    //
    //    // Create a subtract Node.
    //    auto division_op = binary_pw_op_create(afterExponentTensor, afterAddReductionTensor, afterDivisionTensor, divisionDesc);
    //
    //    ops.push_back(std::move(reductionMax_op));
    //    ops.push_back(std::move(subtract_op));
    //    ops.push_back(std::move(exponent_op));
    //    ops.push_back(std::move(reductionAdd_op));
    //    ops.push_back(std::move(division_op));
    //
    //    return afterDivisionTensor;
    //}
    //
    //Softmax::Softmax(cudnnHandle_t& handle, cudnn_frontend::Tensor& inputTensor, void* inputDevPtr, bool verbose)
    //    : Layer(handle, verbose)
    //{
    //    //cudnn_frontend::Tensor flatInputTensor = flattenTensor(inputTensor);
    //    if (verbose)
    //    {
    //        std::cout << "Creating Softmax Layer" << std::endl;
    //    }
    //
    //    auto inputDim = inputTensor.getDim();
    //
    //    Y = std::make_unique<Helpers::Surface<float>>(inputDim[0] * inputDim[1] * inputDim[2], false, 0.0f);
    //
    //    try
    //    {
    //        std::vector<cudnn_frontend::Operation const *> ops;
    //        std::vector<cudnn_frontend::Operation> tmpOps;
    //        yTensor = std::make_unique<cudnn_frontend::Tensor>(createSoftmaxForward(inputDim[0], inputDim[1], inputDim[2], false, CUDNN_DATA_FLOAT, tmpOps, inputTensor));
    //
    //        for (unsigned int i = 0; i < tmpOps.size(); i++) {
    //            ops.push_back(&tmpOps[i]);
    //        }
    //
    //        auto opGraph = cudnn_frontend::OperationGraphBuilder()
    //            .setHandle(mHandle)
    //            .setOperationGraph(ops.size(), ops.data())
    //            .build();
    //
    //        std::cout << opGraph.describe() << std::endl;
    //
    //        plan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));
    //
    //        std::cout << "Plan tag: " << plan->getTag() << std::endl;
    //
    //        workspace_size = plan->getWorkspaceSize();
    //        std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;
    //
    //        if (workspace_size > 0) {
    //            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
    //        }
    //
    //        workspace_size = plan->getWorkspaceSize();
    //        std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;
    //
    //        if (workspace_size > 0) {
    //            checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
    //        }
    //
    //        std::vector<void*> data_ptrs;
    //        data_ptrs.emplace_back(inputDevPtr);
    //        data_ptrs.emplace_back(Y->devPtr);
    //
    //        std::vector<int64_t> uids;
    //        uids.emplace_back(inputTensor.getId());
    //        uids.emplace_back(yTensor->getId());
    //
    //        assert(data_ptrs.size() == uids.size());
    //        int64_t num_ptrs = data_ptrs.size();
    //        std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
    //        variantPack = std::make_unique<cudnn_frontend::VariantPack>(cudnn_frontend::VariantPackBuilder()
    //            .setWorkspacePointer(workspace_ptr)
    //            .setDataPointers(num_ptrs, data_ptrs.data())
    //            .setUids(num_ptrs, uids.data())
    //            .build());
    //        std::cout << "variantPack " << variantPack->describe() << std::endl;
    //    }
    //    catch (cudnn_frontend::cudnnException& e) {
    //        std::cout << "[ERROR] Exception " << e.what() << std::endl;
    //    }
    //}

    //CrossEntropy::CrossEntropy(cudnnHandle_t& handle, cudnn_frontend::Tensor& inputTensor, void* inputDevPtr, bool verbose)
    //    : Layer(handle, verbose)
    //{
    //    _initLoss(inputTensor, inputDevPtr);
    //    _initGrad(inputTensor, inputDevPtr);
    //}

    CrossEntropy::CrossEntropy(cudnnHandle_t& handle, Layer& prevLayer, bool verbose, std::string name)
        : Layer(handle, verbose, std::move(name))
        , mPrevLayer(prevLayer)
    {
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
        printLoss();
        printGrad();

        //checkCudaErr(cudaDeviceSynchronize());
        ////checkCudaErr(cudaMemcpy(P.hostPtr, P.devPtr, size_t(sizeof(P.hostPtr[0])* P.n_elems), cudaMemcpyDeviceToHost));
        //checkCudaErr(cudaMemcpy(J->hostPtr, J->devPtr, size_t(sizeof(J->hostPtr[0]) * J->n_elems), cudaMemcpyDeviceToHost));
        //checkCudaErr(cudaDeviceSynchronize());

        //std::cout << std::format("Loss: {}", -1 * J->hostPtr[0] / mBatchSize) << std::endl;
    }

    void CrossEntropy::printLoss()
    {
        checkCudaErr(cudaDeviceSynchronize());
        checkCudaErr(cudaMemcpy(J->hostPtr, J->devPtr, size_t(sizeof(J->hostPtr[0]) * J->n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaDeviceSynchronize());

        std::cout << std::format("Loss: {}", -1 * J->hostPtr[0] / mBatchSize) << std::endl;
    }

    void CrossEntropy::printGrad()
    {
        // kinda bad. Move to Layer?
        checkCudaErr(cudaDeviceSynchronize());
        checkCudaErr(cudaMemcpy(mPrevLayer.getGradSurface().hostPtr, mPrevLayer.getGradSurface().devPtr, size_t(sizeof(mPrevLayer.getGradSurface().hostPtr[0]) * mPrevLayer.getGradSurface().n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaDeviceSynchronize());

        std::cout << "Cross Entropy derivative:" << std::endl;

        for (size_t b = 0; b < mBatchSize; ++b)
        {
            for (size_t i = 0; i < mNumClasses; ++i)
            {
                std::cout << std::setprecision(3) << mPrevLayer.getGradSurface().hostPtr[b * mNumClasses + i] << " ";
            }
            std::cout << std::endl;
        }
    }

    void CrossEntropy::calculateLoss()
    {

        //TODO: change naming to Loss specific
        if (!plan || !variantPack)
        {
            throw;
        }

        if (mVerbose) std::cout << "calculateLoss" << std::endl;
        try
        {
            cudnnStatus_t status;
            status = cudnnBackendExecute(mHandle, plan->get_raw_desc(), variantPack->get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << cudnnGetErrorString(status) << std::endl;
            }
        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }

    void CrossEntropy::calculateGrad()
    {
        if (!mGradPlan || !mGradVariantPack)
        {
            throw;
        }
        if (mVerbose) std::cout << "calculateGrad" << std::endl;
        try
        {
            cudnnStatus_t status;
            status = cudnnBackendExecute(mHandle, mGradPlan->get_raw_desc(), mGradVariantPack->get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << cudnnGetErrorString(status) << std::endl;
            }
        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }

    //Helpers::Surface<float>& CrossEntropy::getGrad()
    //{
    //    return G.get();
    //}

    //void CrossEntropy::setLabel(const std::vector<int8_t>& labels)
    void CrossEntropy::setLabel(std::span<uint8_t> labels)
    {
        assert(labels.size() == mBatchSize);

        for (size_t b = 0; b < mBatchSize; ++b)
        {
            for (size_t i = 0; i < mNumClasses; ++i)
            {
                if (labels[b] == i)
                {
                    L->hostPtr[b * mNumClasses + i] = 1;
                }
                else
                {
                    L->hostPtr[b * mNumClasses + i] = 0;
                }
                //std::cout << L->hostPtr[b * mNumClasses + i] << " ";
            }
            //std::cout << std::endl;
        }

        checkCudaErr(cudaDeviceSynchronize());
        checkCudaErr(cudaMemcpy(L->devPtr, L->hostPtr, size_t(sizeof(L->hostPtr[0]) * L->n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());
    }

    //void CrossEntropy::_initLoss(cudnn_frontend::Tensor& inputTensor, void* inputDevPtr)
    void CrossEntropy::_initLoss()
    {
        auto& inputTensor = mPrevLayer.getOutputTensor();
        auto inputDim = inputTensor.getDim();
        mBatchSize = inputDim[0];
        assert(inputDim[1] == 1); //sanity check // TODO: multidimensional cross-entropy?
        mNumClasses = inputDim[2]; //we expect it to be flatten // TODO: sanity check?
        int64_t crossEntropyLabelDim[] = { mBatchSize, mNumClasses, 1 }; // we need it to be orthogonal to softmax output, so that we can have a working dot product
        int64_t crossEntropyLabelStride[] = { mNumClasses, 1, 1 }; // does that makes sense?
        int64_t matmulDim[] = { mBatchSize, 1, 1 }; // magic numbers are justified here?
        int64_t matmulStride[] = { 1,1,1 };

        P = std::make_unique<Helpers::Surface<float>>(matmulDim[0] * matmulDim[1] * matmulDim[2], false, 0.0f);

        //int64_t labelStride[] = { mNumClasses, mNumClasses, 1 }; // does this?

        int64_t lossDim[] = { 1, 1, 1 }; // I think it should be that since we need to have an average across dims including the batch
        int64_t lossStride[] = { 1,1,1 }; // not sure if this makes sense
        J = std::make_unique<Helpers::Surface<float>>(lossDim[0] * lossDim[1] * lossDim[2], false, 0.0f);

        L = std::make_unique<Helpers::Surface<float>>(mBatchSize * mNumClasses, false, 0.0f);

        // HYPERPARAMETERS
        constexpr int64_t alignment = 16; //16
        const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        constexpr int64_t nbDims = 3;

        using namespace cudnn_frontend;

        try
        {
            //auto xTensor = cudnn_frontend::TensorBuilder()
            //    .setDim(nbDims, inputDim)
            //    .setStride(nbDims, crossEntropyLabelStride) // not sure, we'll see
            //    .setId(getTensorId())
            //    .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
            //    .setDataType(dataType)
            //    .build();
            //std::cout << xTensor.describe() << std::endl;

            auto crossEntropyLabelTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment) // this needs to be a function
                .setDim(3, crossEntropyLabelDim)
                .setStride(3, crossEntropyLabelStride)
                .setId(getTensorId())
                .build();
            if (mVerbose) std::cout << crossEntropyLabelTensor.describe() << std::endl;

            auto afterProductTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment)
                .setDim(3, matmulDim)
                .setStride(3, matmulStride)
                .setId(getTensorId())
                //.setVirtual(true) // apparently it cannot be virtual 
                .build();
            if (mVerbose) std::cout << afterProductTensor.describe() << std::endl;

            auto afterLogTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment)
                .setDim(3, matmulDim)
                .setStride(3, matmulStride)
                .setId(getTensorId())
                .setVirtual(true)
                .build();
            if (mVerbose) std::cout << afterLogTensor.describe() << std::endl;

            // the output
            auto lossTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment)
                .setDim(3, lossDim)
                .setStride(3, lossStride)
                .setId(getTensorId())
                .build();
            if (mVerbose) std::cout << lossTensor.describe() << std::endl;

            // loss ops

            auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
            // Create a matmul Node
            if (mVerbose) std::cout << "Matmul" << std::endl;
            auto matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                .setaMatDesc(inputTensor)
                .setbMatDesc(crossEntropyLabelTensor)
                .setcMatDesc(afterProductTensor)
                .setmatmulDesc(matmulDesc)
                .build();
            if (mVerbose) std::cout << matmul_op.describe() << std::endl;

            auto pwLogDesc = PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_LOG)
                .setComputeType(dataType)
                .build();
            if (mVerbose) std::cout << pwLogDesc.describe() << std::endl;

            if (mVerbose) std::cout << "pwLog_op" << std::endl;
            auto pwLog_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(matmul_op.getOutputTensor())
                .setyDesc(afterLogTensor)
                .setpwDesc(pwLogDesc)
                .build();
            if (mVerbose) std::cout << pwLog_op.describe() << std::endl;

            if (mVerbose) std::cout << "avgReductionDesc" << std::endl;
            auto avgReductionDesc = ReductionDescBuilder()
                .setComputeType(dataType)
                .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                .build();
            if (mVerbose) std::cout << avgReductionDesc.describe() << std::endl;

            if (mVerbose) std::cout << "avgReduction_op" << std::endl;
            auto avgReduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                .setxDesc(pwLog_op.getOutputTensor())
                .setyDesc(lossTensor)
                .setreductionDesc(avgReductionDesc)
                .build();
            if (mVerbose) std::cout << avgReduction_op.describe() << std::endl;

            std::array<cudnn_frontend::Operation const*, 3> ops = { &matmul_op, &pwLog_op, &avgReduction_op };

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(mHandle)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            if (mVerbose) std::cout << opGraph.describe() << std::endl;

            plan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));

            if (mVerbose) std::cout << "Plan tag: " << plan->getTag() << std::endl;

            workspace_size = plan->getWorkspaceSize();
            if (mVerbose) std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }

            std::vector<void*> data_ptrs;
            //data_ptrs.emplace_back(inputDevPtr);
            data_ptrs.emplace_back(mPrevLayer.getOutputSurface().devPtr);
            data_ptrs.emplace_back(L->devPtr);
            data_ptrs.emplace_back(P->devPtr);
            data_ptrs.emplace_back(J->devPtr);

            std::vector<int64_t> uids;
            //uids.emplace_back(inputTensor.getId());
            uids.emplace_back(mPrevLayer.getOutputTensor().getId());
            uids.emplace_back(crossEntropyLabelTensor.getId());
            uids.emplace_back(afterProductTensor.getId());
            uids.emplace_back(lossTensor.getId());

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
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }

    void CrossEntropy::_initGrad()
    {
        auto& inputTensor = mPrevLayer.getOutputTensor();
        // TODO: Proper defaults
        constexpr int64_t alignment = 16;
        const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        constexpr int64_t nbDims = 3;

        using namespace cudnn_frontend;

        auto inputDim = inputTensor.getDim();
        assert(inputTensor.getDimCount() == nbDims); // sanity check, remove if ever allow different dimensiality
        int64_t gradDim[] = { inputDim[0], inputDim[1], inputDim[2] };
        int64_t gradStride[] = { inputDim[1] * inputDim[2], 1, 1 };

        //G = std::make_unique<Helpers::Surface<float>>(inputDim[0] * inputDim[1] * inputDim[2], false, 0.0f);

        try
        {
            auto labelTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment)
                .setId(getTensorId())
                .setDim(nbDims, gradDim)
                .setStride(nbDims, gradStride)
                .build();
            if (mVerbose) std::cout << labelTensor.describe() << std::endl;

            auto gradTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment)
                .setId(getTensorId())
                .setDim(nbDims, gradDim)
                .setStride(nbDims, gradStride)
                .build();
            if (mVerbose) std::cout << gradTensor.describe() << std::endl;

            auto subDesc = PointWiseDescBuilder()
                .setComputeType(dataType)
                .setMode(CUDNN_POINTWISE_SUB)
                .build();
            if (mVerbose) std::cout << subDesc.describe() << std::endl;

            auto pw_sub_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(inputTensor)
                .setbDesc(labelTensor)
                .setyDesc(gradTensor)
                .setpwDesc(subDesc)
                .build();
            if (mVerbose) std::cout << pw_sub_op.describe() << std::endl;


            // TODO: this block needs to be a function...
            std::vector<cudnn_frontend::Operation const*> ops = { &pw_sub_op };

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(mHandle)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            if (mVerbose) std::cout << opGraph.describe() << std::endl;

            mGradPlan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), mHandle));

            if (mVerbose) std::cout << "Plan tag: " << mGradPlan->getTag() << std::endl;

            mGradWorkspaceSize = mGradPlan->getWorkspaceSize();
            if (mVerbose) std::cout << mGradPlan->describe() << " requires workspace " << mGradWorkspaceSize << std::endl;

            if (mGradWorkspaceSize > 0) {
                checkCudaErr(cudaMalloc(&mGradWorkspacePtr, (size_t)mGradWorkspaceSize));
            }

            std::vector<void*> data_ptrs;
            //data_ptrs.emplace_back(inputDevPtr);
            data_ptrs.emplace_back(mPrevLayer.getOutputSurface().devPtr);
            data_ptrs.emplace_back(L->devPtr);
            //data_ptrs.emplace_back(G->devPtr);
            data_ptrs.emplace_back(mPrevLayer.getGradSurface().devPtr);

            std::vector<int64_t> uids;
            uids.emplace_back(inputTensor.getId());
            uids.emplace_back(labelTensor.getId());
            uids.emplace_back(gradTensor.getId());

            assert(data_ptrs.size() == uids.size());
            int64_t num_ptrs = data_ptrs.size();
            if (mVerbose) std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
            mGradVariantPack = std::make_unique<cudnn_frontend::VariantPack>(cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(mGradWorkspacePtr)
                .setDataPointers(num_ptrs, data_ptrs.data())
                .setUids(num_ptrs, uids.data())
                .build());
            if (mVerbose) std::cout << "Grad variantPack " << mGradVariantPack->describe() << std::endl;


        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }
    }
}
