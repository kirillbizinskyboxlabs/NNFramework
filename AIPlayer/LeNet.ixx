module;
#pragma once
//#include <cudnn_frontend.h>
#include "LeNetHelpers.h"
#include "LenetLayers.h"
#include <cudnn.h>
#include "NeuralNetwork.h"

//namespace 
//{
//    auto heurgen_method = [](cudnn_frontend::OperationGraph& opGraph) -> cudnn_frontend::EngineConfigList {
//        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
//            .setOperationGraph(opGraph)
//            .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
//            .build();
//        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
//
//        auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
//        cudnn_frontend::EngineConfigList filtered_configs;
//        cudnn_frontend::filter(engine_configs, filtered_configs, Helpers::allowAll);
//        return filtered_configs;
//    };
//}

export module LeNet;

import <iostream>;
import <array>;
import <format>;

//import LeNetHelpers;

import MNISTData;


namespace LeNet
{
    void display(float* image, uint8_t label, size_t size)
    {
        //std::cout << "Label: " << static_cast<int>(label) << "\n";
        std::cout << std::format("Label: {} Size {}", static_cast<int>(label), size) << std::endl;
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

    void display_flat(float* image, uint8_t label, size_t size)
    {
        //std::cout << "Label: " << static_cast<int>(label) << "\n";
        std::cout << std::format("Label: {} Size {}", static_cast<int>(label), size) << std::endl;
        for (size_t i = 0; i < size; ++i)
        {
            std::cout << std::setprecision(3) << image[i] << " ";
        }

        std::cout << "\n";
    }

    void display(float* image, uint8_t label)
    {
        std::cout << "Label: " << static_cast<int>(label) << "\n";
        for (size_t r = 0; r < 28; ++r)
        {
            for (size_t c = 0; c < 28; ++c)
            {
                std::cout << std::setprecision(3) << image[r * 28 + c] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "\n";
    }

    void display(int8_t* image, uint8_t label)
    {
        std::cout << "Label: " << static_cast<int>(label) << "\n";
        for (size_t r = 0; r < 28; ++r)
        {
            for (size_t c = 0; c < 28; ++c)
            {
                std::cout << std::setw(3) << image[r * 28 + c] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "\n";
    }

    void display(float* image, int64_t num_elements)
    {
        std::cout << "Num elem: " << num_elements << std::endl;
        for (size_t r = 0; r < sqrt(num_elements); ++r)
        {
            for (size_t c = 0; c < sqrt(num_elements); ++c)
            {
                std::cout << std::setprecision(3) << image[r * static_cast<int>(sqrt(num_elements)) + c] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "\n";
    }

    void LenetForward_v_10();
    void LenetForward_v_9();
    void LenetForward_v_8();
    void LenetForward_v_7();
    void LenetForward_v_6();
    void LenetForward_v_5();
    void LenetForward_v_4();
    void LenetForward_v_3();

    void SoftmaxTest();
    void SoftmaxBackendTest();
    void SoftmaxBackendTestv2();
    void CrossEntropyTest();

    export void LenetForward()
    {
        //SoftmaxBackendTestv2();
        //LenetForward_v_8();
        LenetForward_v_10();
        //LenetForward_v_9();
        //CrossEntropyTest();
    }

    void LenetForward_v_10()
    {
        std::cout << "LeNet Test v1.0" << std::endl;

        // Hyperparameters?
        constexpr int64_t batchSize = 512;
        constexpr int64_t inputH = 28;
        constexpr int64_t inputW = 28;
        constexpr int64_t C1Features = 32;
        constexpr int64_t C1KernelSize = 5;
        constexpr int64_t C1Padding = 2;
        constexpr int64_t C3Features = 64;
        constexpr int64_t C3KernelSize = 5;
        constexpr int64_t C3Padding = 0;
        constexpr int64_t C5Features = 1024;
        constexpr int64_t C5KernelSize = 5;
        constexpr int64_t C5Padding = 0;
        constexpr int64_t FC6OutputSize = 84;
        constexpr int64_t FC7OutputSize = 10;

        bool verbose = false;

        MNISTDataHolder dh;
        dh.initialize();
        //auto [image, label] = dh.getNextTrain();
        //auto [images, labels] = dh.getNextNTrain(batchSize);
        auto [rows, cols] = dh.getDimensions();

        std::vector<size_t> dims = { 1, rows, cols };

        NeuralNetwork nn(batchSize, dims.size(), dims.data(), NeuralNetwork::VERBOSITY::MIN);
        nn.addConvBiasAct(C1KernelSize, C1Features, C1Padding, verbose, "C1");
        nn.addPool(verbose, "S2");
        nn.addConvBiasAct(C3KernelSize, C3Features, C3Padding, verbose, "C3");
        nn.addPool(verbose, "S4");
        nn.addConvBiasAct(C5KernelSize, C5Features, 0, verbose, "FC5");
        //nn.addConvBiasAct(1, FC6OutputSize, 0, verbose, "FC6");
        nn.addConvBiasAct(1, FC7OutputSize, 0, verbose, "FC7");
        nn.addSoftmax(verbose);
        nn.addCrossEntropy(verbose);


        size_t epoch_num = 200;
        size_t iter_num = epoch_num * (dh.getTrainSize() / batchSize);


        while(iter_num--)
        {
            dh.loadData(batchSize, nn.getInputDataPtr(), nn.getLabelDataPtr());
            nn.syncLabel();

            nn.train();

            if (iter_num % (dh.getTrainSize()/ batchSize) == 0)
            {
                std::cout << std::format("Iter {} ", iter_num);
                nn.printLoss();
            }
        }
    }


    void CrossEntropyTest()
    {
        using namespace Helpers;
        constexpr int64_t batchSize = 16;
        constexpr int64_t numClasses = 10;

        int64_t crossEntropyLabelDim[] = { batchSize, numClasses, 1 }; // we need it to be orthogonal to softmax output, so that we can have a working dot product
        int64_t crossEntropyLabelStride[] = { numClasses, 1, 1 }; // does that makes sense?
        int64_t matmulDim[] = { batchSize, 1, 1 }; // magic numbers are justified here?
        int64_t matmulStride[] = { 1,1,1 };

        Surface<float> P(matmulDim[0] * matmulDim[1] * matmulDim[2], false, 0.0f);

        int64_t labelStride[] = { numClasses, numClasses, 1 }; // does this?

        int64_t lossDim[] = { 1, 1, 1 }; // I think it should be that since we need to have an average across dims including the batch
        int64_t lossStride[] = { 1,1,1 }; // not sure if this makes sense
        Helpers::Surface<float> J(lossDim[0] * lossDim[1] * lossDim[2], false, 0.0f);
        //Helpers::Surface<float> M(lossDim[0] * lossDim[1] * lossDim[2], false, 0.0f); // maybe not needed if average works as expected

        int64_t xTensorDim[] = { batchSize, 1, numClasses };
        Surface<float> X(batchSize * numClasses, false, 0.5f); // random input, maybe I shall softmax it first?
        Surface<float> L(batchSize * numClasses, false, 0.0f);

        for (size_t i = 0; i < batchSize; ++i)
        {
            L.hostPtr[numClasses * i + 5] = 1;
        }

        //for (size_t b = 0; b < batchSize; ++b)
        //{
        //    for (size_t i = 0; i < numClasses; ++i)
        //    {
        //        std::cout << std::setprecision(3) << X.hostPtr[b * numClasses + i] << " ";
        //    }

        //    std::cout << std::endl;
        //}

        //for (size_t b = 0; b < batchSize; ++b)
        //{
        //    for (size_t i = 0; i < numClasses; ++i)
        //    {
        //        std::cout << std::setprecision(3) << L.hostPtr[b * numClasses + i] << " ";
        //    }

        //    std::cout << std::endl;
        //}

        //checkCudaErr(cudaMemcpy(X.devPtr, X.hostPtr, size_t(sizeof(X.hostPtr[0]) * X.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(L.devPtr, L.hostPtr, size_t(sizeof(L.hostPtr[0]) * L.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle;
        checkCudnnErr(cudnnCreate(&handle));

        // HYPERPARAMETERS
        constexpr int64_t alignment = 16; //16
        //const cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
        const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        constexpr int64_t nbDims = 3;

        using namespace cudnn_frontend;

        try
        {
            auto xTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, xTensorDim)
                .setStride(nbDims, crossEntropyLabelStride) // not sure, we'll see
                .setId('x')
                .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();
            std::cout << xTensor.describe() << std::endl;

            auto crossEntropyLabelTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment) // this needs to be a function
                .setDim(3, crossEntropyLabelDim)
                .setStride(3, crossEntropyLabelStride)
                .setId('y')
                .build();
            std::cout << crossEntropyLabelTensor.describe() << std::endl;

            auto afterProductTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment)
                .setDim(3, matmulDim)
                .setStride(3, matmulStride)
                .setId('P') // as greek capital Pi dinoting product
                //.setVirtual(true)
                .build();
            std::cout << afterProductTensor.describe() << std::endl;

            auto afterLogTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment)
                .setDim(3, matmulDim)
                .setStride(3, matmulStride)
                .setId('L')
                .setVirtual(true)
                .build();
            std::cout << afterLogTensor.describe() << std::endl;

            // the output
            auto lossTensor = TensorBuilder()
                .setDataType(dataType)
                .setAlignment(alignment)
                .setDim(3, lossDim)
                .setStride(3, lossStride)
                .setId('j')
                .build();
            std::cout << lossTensor.describe() << std::endl;

            // loss ops

            auto matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
            // Create a matmul Node
            std::cout << "Matmul" << std::endl;
            auto matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                .setaMatDesc(xTensor)
                .setbMatDesc(crossEntropyLabelTensor)
                .setcMatDesc(afterProductTensor)
                .setmatmulDesc(matmulDesc)
                .build();
            std::cout << matmul_op.describe() << std::endl;

            auto pwLogDesc = PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_LOG)
                .setComputeType(dataType)
                .build();
            std::cout << pwLogDesc.describe() << std::endl;

            std::cout << "pwLog_op" << std::endl;
            auto pwLog_op = OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(matmul_op.getOutputTensor())
                .setyDesc(afterLogTensor)
                .setpwDesc(pwLogDesc)
                .build();
            std::cout << pwLog_op.describe() << std::endl;

            std::cout << "avgReductionDesc" << std::endl;
            auto avgReductionDesc = ReductionDescBuilder()
                .setComputeType(dataType)
                .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                .build();
            std::cout << avgReductionDesc.describe() << std::endl;

            std::cout << "avgReduction_op" << std::endl;
            auto avgReduction_op = OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                .setxDesc(pwLog_op.getOutputTensor())
                .setyDesc(lossTensor)
                .setreductionDesc(avgReductionDesc)
                .build();
            std::cout << avgReduction_op.describe() << std::endl;

            std::array<cudnn_frontend::Operation const*, 3> ops = { &matmul_op, &pwLog_op, &avgReduction_op };

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            std::cout << opGraph.describe() << std::endl;

            auto plan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle));

            std::cout << "Plan tag: " << plan->getTag() << std::endl;

            auto workspace_size = plan->getWorkspaceSize();
            void* workspace_ptr = nullptr;
            std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }

            std::vector<void*> data_ptrs;
            data_ptrs.emplace_back(X.devPtr);
            data_ptrs.emplace_back(L.devPtr);
            data_ptrs.emplace_back(P.devPtr);
            data_ptrs.emplace_back(J.devPtr);

            std::vector<int64_t> uids;
            uids.emplace_back(xTensor.getId());
            uids.emplace_back(crossEntropyLabelTensor.getId());
            uids.emplace_back(afterProductTensor.getId());
            uids.emplace_back(lossTensor.getId());

            assert(data_ptrs.size() == uids.size());
            int64_t num_ptrs = data_ptrs.size();
            std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
            auto variantPack = std::make_unique<cudnn_frontend::VariantPack>(cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                .setDataPointers(num_ptrs, data_ptrs.data())
                .setUids(num_ptrs, uids.data())
                .build());
            std::cout << "variantPack " << variantPack->describe() << std::endl;

            cudnnStatus_t status;
            status = cudnnBackendExecute(handle, plan->get_raw_desc(), variantPack->get_raw_desc());

            std::cout << cudnnGetErrorString(status) << std::endl;

        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }

        checkCudaErr(cudaDeviceSynchronize());
        //checkCudaErr(cudaMemcpy(P.hostPtr, P.devPtr, size_t(sizeof(P.hostPtr[0])* P.n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(J.hostPtr, J.devPtr, size_t(sizeof(J.hostPtr[0]) * J.n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaDeviceSynchronize());

        std::cout << std::format("Loss: {}", J.hostPtr[0]/batchSize) << std::endl;

        //display_flat(P.hostPtr, 0, P.n_elems);

        if (handle) cudnnDestroy(handle);
    }

    void LenetForward_v_9()
    {
        std::cout << "Less rigid forward LeNet Test v0.9" << std::endl;

        // Hyperparameters?
        constexpr int64_t batchSize = 2;
        constexpr int64_t inputH = 28;
        constexpr int64_t inputW = 28;
        constexpr int64_t C1Features = 6;
        constexpr int64_t C1KernelSize = 5;
        constexpr int64_t C1Padding = 2;
        constexpr int64_t C3Features = 16;
        constexpr int64_t C3KernelSize = 5;
        constexpr int64_t C3Padding = 0;
        constexpr int64_t C5Features = 120;
        constexpr int64_t C5KernelSize = 5;
        constexpr int64_t C5Padding = 0;
        constexpr int64_t FC6OutputSize = 84;
        constexpr int64_t FC7OutputSize = 10;

        bool verbose = true;

        MNISTDataHolder dh;
        dh.initialize();
        //auto [image, label] = dh.getNextTrain();
        auto [images, labels] = dh.getNextNTrain(batchSize);
        auto [rows, cols] = dh.getDimensions();

        // Input
        int64_t xTensorDim[] = { batchSize, 1, inputH, inputW }; // input
        Helpers::Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3]);

        int64_t surfaceDataPtr = 0;
        for (auto&& image : images)
        {
            for (auto&& pixel : image)
            {
                X.hostPtr[surfaceDataPtr++] = static_cast<float>(pixel) / 256;
                //std::cout << std::setw(3) << static_cast<int>(pixel) << " " << static_cast<float>(pixel) << " ";
                //if (surfaceDataPtr > 1 && ((surfaceDataPtr + 1) % cols == 0))
                //{
                //    std::cout << std::endl;
                //}
            }
            //std::cout << std::endl << std::endl;
        }

        checkCudaErr(cudaMemcpy(X.devPtr, X.hostPtr, size_t(sizeof(X.hostPtr[0]) * X.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle;
        checkCudnnErr(cudnnCreate(&handle));

        // HYPERPARAMETERS
        constexpr int64_t alignment = 16; //16
        const cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
        const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
        constexpr int64_t nbDims = 4;

        try
        {
            int64_t stride[nbDims];
            Helpers::generateStrides(xTensorDim, stride, nbDims, tensorFormat);
            auto xTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, xTensorDim)
                .setStride(nbDims, stride)
                .setId('x')
                .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();


            ConvBiasAct C1(handle, xTensorDim, C1KernelSize, C1Features, xTensor, X.devPtr, C1Padding, verbose);
            Pool S2(handle, C1.getOutputTensor(), C1.getOutputSurface().devPtr, verbose);
            ConvBiasAct C3(handle, S2.getOutputTensor().getDim(), C3KernelSize, C3Features, S2.getOutputTensor(), S2.getOutputSurface().devPtr, C3Padding, verbose);
            Pool S4(handle, C3.getOutputTensor(), C3.getOutputSurface().devPtr, verbose);
            ConvBiasAct C5(handle, S4.getOutputTensor().getDim(), C5KernelSize, C5Features, S4.getOutputTensor(), S4.getOutputSurface().devPtr, C5Padding, verbose);
            //FC FC6(handle, C5.getOutputTensor(), C5.getOutputSurface().devPtr, FC6OutputSize, verbose);
            //FC FC7(handle, FC6.getOutputTensor(), FC6.getOutputSurface().devPtr, FC7OutputSize, verbose);
            FC FC6(handle, C5, FC6OutputSize, verbose, "FC6");
            FC FC7(handle, FC6, FC7OutputSize, true, "FC7");
            //Softmax SM8(handle, FC7.getOutputTensor(), FC7.getOutputSurface(), true);
            Softmax SM8(handle, FC7, verbose, "Softmax");
            //CrossEntropy CE(handle, SM8.getOutputTensor(), SM8.getOutputSurface().devPtr, true);
            CrossEntropy CE(handle, SM8, verbose, "CrossEnropy");

            CE.setLabel(labels);

            C1.executeInference();
            S2.executeInference();
            C3.executeInference();
            S4.executeInference();
            C5.executeInference();
            FC6.executeInference();
            FC7.executeInference();
            SM8.executeInference();
            CE.calculateLoss();
            CE.calculateGrad();

            SM8.backpropagate();

            //C1.printOutput();
            //S2.printOutput();
            //C3.printOutput();
            //S4.printOutput();
            //C5.printOutput();
            //FC6.printOutput();
            FC7.printOutput();
            SM8.printOutput();
            CE.printOutput();
            FC7.printGrad();

            FC7.printBias();
            FC7.printWeights();

            FC7.backpropagate();
            FC7.printBias();
            FC7.printWeights();

            //auto& FC7.get
        }
        catch (...)
        {
            std::cout << "Error during execution" << std::endl;
        }

        //CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;

        if (handle) cudnnDestroy(handle);
    }


    void SoftmaxBackendTestv2()
    {
        using namespace Helpers;
        constexpr int64_t nbDims = 3;
        constexpr int64_t batchSize = 3;
        int64_t inputDim[] = { batchSize,1,10 };
        int64_t strideA[] = { inputDim[1] * inputDim[2], inputDim[2], 1 };

        Surface<float> X(inputDim[0] * inputDim[1] * inputDim[2]);

        cudnnHandle_t cudnnHandle;
        cudnnCreate(&cudnnHandle);

        auto inputTensor = cudnn_frontend::TensorBuilder()
            .setDim(nbDims, inputDim)
            .setStride(nbDims, strideA)
            .setId('x')
            .setAlignment(16)
            .setDataType(CUDNN_DATA_FLOAT)
            .build();

        //Softmax SM(cudnnHandle, inputTensor, X, true);

        //SM.executeInference();

        //SM.printOutput();
    }

    void SoftmaxBackendTest()
    {
        using namespace Helpers;
        int64_t inputDim[] = { 1,1,10 };

        Surface<float> X(10);
        Surface<float> Y(10, false, 0.0f);

        //int m = 5, c = 4, numChannels = 1;
        int m = 10, c = 1, numChannels = 1;

        //srand(time(NULL));
        //double* fcLayer = (double*)malloc(m * c * sizeof(double));
        //for (int i = 0; i < m; i++) {
        //    double def = rand() % 25;
        //    for (int c_idx = 0; c_idx < c; c_idx++) {
        //        int offset = i * c + c_idx;
        //        fcLayer[offset] = def;
        //    }
        //}
        //printf("FC LAYER:\n");
        //printMatrix(fcLayer, c, m);

        //double* d_fcLayer;
        //cudaMalloc((void**)&d_fcLayer, m * c * sizeof(double));
        //cudaMemcpy(d_fcLayer, fcLayer, m * c * sizeof(double), cudaMemcpyHostToDevice);

        //double* d_softmaxData;
        //cudaMalloc((void**)&d_softmaxData, m * c * sizeof(double));

        cudnnHandle_t cudnnHandle;
        cudnnCreate(&cudnnHandle);

        // softmaxForward(n, c, h, w, dstData, &srcData);
        //cudnnTensor4dDescriptor_t srcTensorDesc, sftTensorDesc;
        //cudnnCreateTensor4dDescriptor(&srcTensorDesc);
        //cudnnCreateTensor4dDescriptor(&sftTensorDesc);

        cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;

        cudnnCreateTensorDescriptor(&srcTensorDesc);
        cudnnCreateTensorDescriptor(&sftTensorDesc);

        //cudnnSetTensor4dDescriptor

        int nbDims = 3;
        int dimA[] = { 1,1,10 };
        int strideA[] = { 1,1,1 };

        float alpha = 1.0f;
        float beta = 0.0f;

        //cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        //    1, 1, 1, 10);
        //cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        //    1, 1, 1, 10);

        cudnnSetTensorNdDescriptor(srcTensorDesc,
            CUDNN_DATA_FLOAT,
            nbDims,
            dimA,
            strideA);
        cudnnSetTensorNdDescriptor(sftTensorDesc,
            CUDNN_DATA_FLOAT,
            nbDims,
            dimA,
            strideA);

        auto status = cudnnSoftmaxForward(cudnnHandle, 
                            CUDNN_SOFTMAX_ACCURATE, 
                            CUDNN_SOFTMAX_MODE_INSTANCE,
                            &alpha,
                            srcTensorDesc,
                            X.devPtr,
                            &beta,
                            sftTensorDesc, 
                            Y.devPtr);

        std::cout << cudnnGetErrorString(status) << std::endl;

        cudaDeviceSynchronize();

        // Copy back
        //double* result = (double*)malloc(m * c * sizeof(double));
        //cudaMemcpy(result, d_softmaxData, m * c * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Y.hostPtr, Y.devPtr, Y.n_elems * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        display_flat(X.hostPtr, 0, 10);
        display_flat(Y.hostPtr, 0, 10);

        //cudnnSoftmaxBackward()

        //// Log
        //printf("SOFTMAX:\n");
        //printMatrix(result, c, m);

        //// Try backward
        //cudnnTensor4dDescriptor_t diffTensorDesc;
        //cudnnCreateTensor4dDescriptor(&diffTensorDesc);
        //cudnnSetTensor4dDescriptor(diffTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        //    m, c, 1, 1);

        //double* d_gradData;
        //cudaMalloc((void**)&d_gradData, m * c * sizeof(double));

        //double* diffData = makeDiffData(m, c);
        //double* d_diffData;
        //cudaMalloc((void**)&d_diffData, m * c * sizeof(double));
        //cudaMemcpy(d_diffData, diffData, m * c * sizeof(double), cudaMemcpyHostToDevice);
        //cudaDeviceSynchronize();

        //cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        //    srcTensorDesc, d_softmaxData, diffTensorDesc, d_diffData, sftTensorDesc, d_gradData);
        //cudaDeviceSynchronize();

        //// Copy back
        //double* result_backward = (double*)malloc(m * c * sizeof(double));
        //cudaMemcpy(result_backward, d_gradData, m * c * sizeof(double), cudaMemcpyDeviceToHost);
        //cudaDeviceSynchronize();

        //// Log
        //printf("GRADIENT:\n");
        //printMatrix(result_backward, c, m);

        //// Destruct
        //free(result);
        //free(diffData);
        //free(result_backward);
        //free(fcLayer);

        //cudnnDestroyTensor4dDescriptor(srcTensorDesc);
        //cudnnDestroyTensor4dDescriptor(sftTensorDesc);
        //cudnnDestroyTensor4dDescriptor(diffTensorDesc);
        //cudaFree(d_fcLayer);
        //cudaFree(d_softmaxData);
        //cudaFree(d_gradData);
        //cudaFree(d_diffData);
        cudnnDestroy(cudnnHandle);
    }

    void SoftmaxTest()
    {
        using namespace Helpers;
        int64_t softmaxDim[] = { 1,1,10 };
        int64_t stride[] = { 10,10,1 };

        int64_t afterReductionDim[] = { 1, 1, 1 };
        int64_t afterReductionStride[] = { 1, 1, 1 };

        Surface<float> X(10);
        Surface<float> Y(10, false, 0.0f);
        Surface<float> Z(1, false, 1.0f);
        Surface<float> E(1, false, 1.0f);

        cudnnHandle_t handle;
        checkCudnnErr(cudnnCreate(&handle));

        try
        {
            int nbDims = 3;
            auto inputTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, softmaxDim)
                .setStride(nbDims, stride)
                .setId('x')
                .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                .setDataType(CUDNN_DATA_FLOAT)
                //.setVirtual(is_virtual)
                //.setByValue(is_value)
                .build();
            std::cout << inputTensor.describe() << std::endl;

            auto afterMaxReductionTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, afterReductionDim)
                .setStride(nbDims, afterReductionStride)
                .setId('m')
                .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                .setDataType(CUDNN_DATA_FLOAT)
                //.setVirtual(true)
                //.setByValue(is_value)
                .build();
            std::cout << afterMaxReductionTensor.describe() << std::endl;

            auto afterSubtractionTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, softmaxDim)
                .setStride(nbDims, stride)
                .setId('S')
                .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                .setDataType(CUDNN_DATA_FLOAT)
                .setVirtual(true)
                //.setByValue(is_value)
                .build();
            std::cout << afterSubtractionTensor.describe() << std::endl;

            auto afterExponentTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, softmaxDim)
                .setStride(nbDims, stride)
                .setId('e')
                .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                .setDataType(CUDNN_DATA_FLOAT)
                .setVirtual(true)
                //.setByValue(is_value)
                .build();
            std::cout << afterExponentTensor.describe() << std::endl;

            auto afterAddReductionTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, afterReductionDim)
                .setStride(nbDims, afterReductionStride)
                .setId('E')
                .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                .setDataType(CUDNN_DATA_FLOAT)
                .setVirtual(true)
                //.setByValue(is_value)
                .build();
            std::cout << afterAddReductionTensor.describe() << std::endl;

            //// divide (e/ sum(e))
            //std::cout << "afterDivisionTensor" << std::endl;
            //auto afterDivisionTensor = cudnn_frontend::TensorBuilder()
            //    .setDim(nbDims, afterBMM1_dim)
            //    .setStride(nbDims, afterBMM1_stride)
            //    .setId(softmaxOutputName)
            //    .setAlignment(16) // 16B alignment is needed to run a tensor core engine
            //    .setDataType(softmaxOutputType)
            //    .setVirtual(softmax_output_virtual)
            //    .setByValue(false)
            //    //.setReorderType(cudnn_frontend::cudnnBackendTensorReordering_t::CUDNN_TENSOR_REORDERING_F16x16) // do I need it?
            //    .build();
            //std::cout << afterDivisionTensor.describe() << std::endl;

            auto outputTensor = cudnn_frontend::TensorBuilder()
                .setDim(nbDims, softmaxDim)
                .setStride(nbDims, stride)
                .setId('y')
                .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                .setDataType(CUDNN_DATA_FLOAT)
                //.setVirtual(is_virtual)
                //.setByValue(is_value)
                .build();
            std::cout << outputTensor.describe() << std::endl;


            //std::cout << "reductionMaxDesc" << std::endl;
            //auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
            //    .setComputeType(CUDNN_DATA_FLOAT)
            //    .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
            //    .build();
            //std::cout << reductionMaxDesc.describe() << std::endl;

            //// Create a reduction max Node.
            //std::cout << "reductionMax_op" << std::endl;
            //auto reductionMax_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
            //    .setxDesc(inputTensor)
            //    .setyDesc(afterMaxReductionTensor)
            //    .setreductionDesc(reductionMaxDesc)
            //    .build();
            //std::cout << reductionMax_op.describe() << std::endl;

            // Define the subtract descriptor
            auto pw_sub_desc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_SUB)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();

            std::cout << pw_sub_desc.describe() << std::endl;

            // Create a subtract Node.
            auto pw_sub_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(inputTensor)
                .setbDesc(afterMaxReductionTensor)
                .setyDesc(afterSubtractionTensor)
                .setpwDesc(pw_sub_desc)
                .build();
            std::cout << pw_sub_op.describe() << std::endl;

            // Define the exponent descriptor
            auto pw_exp_desc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_EXP)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << pw_exp_desc.describe() << std::endl;

            // Create a exponent Node.
            auto pw_exp_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(afterSubtractionTensor)
                .setyDesc(afterExponentTensor)
                .setpwDesc(pw_exp_desc)
                .build();
            std::cout << pw_exp_op.describe() << std::endl;

            // Define the sum reduction descriptor
            auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                .setComputeType(CUDNN_DATA_FLOAT)
                .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                .build();
            std::cout << reductionAddDesc.describe() << std::endl;

            // Create a sum reduction Node.
            auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                .setxDesc(afterExponentTensor) //afterExponentTensor
                .setyDesc(afterAddReductionTensor)
                .setreductionDesc(reductionAddDesc)
                .build();

            std::cout << reductionAdd_op.describe() << std::endl;

            // Define the div descriptor
            auto pw_div_desc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_DIV)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << pw_div_desc.describe() << std::endl;

            // Create a div Node.
            auto pw_div_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(afterExponentTensor)
                .setbDesc(afterAddReductionTensor)
                .setyDesc(outputTensor)
                .setpwDesc(pw_div_desc)
                .build();
            std::cout << pw_div_op.describe() << std::endl;

            CUDNN_VERSION;


            std::vector<cudnn_frontend::Operation const*> ops;
            //ops.emplace_back(&reductionMax_op);
            ops.emplace_back(&pw_sub_op);
            ops.emplace_back(&pw_exp_op);
            ops.emplace_back(&reductionAdd_op);
            ops.emplace_back(&pw_div_op);

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            std::cout << opGraph.describe() << std::endl;

            auto plan = std::make_unique<cudnn_frontend::ExecutionPlan>(Helpers::get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle));

            std::cout << "Plan tag: " << plan->getTag() << std::endl;

            auto workspace_size = plan->getWorkspaceSize();
            void* workspace_ptr = nullptr;
            std::cout << plan->describe() << " requires workspace " << workspace_size << std::endl;

            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }

            std::vector<void*> data_ptrs;
            data_ptrs.emplace_back(X.devPtr);
            data_ptrs.emplace_back(Z.devPtr);
            //data_ptrs.emplace_back(E.devPtr);
            data_ptrs.emplace_back(Y.devPtr);

            std::vector<int64_t> uids;
            uids.emplace_back(inputTensor.getId());
            uids.emplace_back(afterMaxReductionTensor.getId());
            //uids.emplace_back(afterAddReductionTensor.getId());
            uids.emplace_back(outputTensor.getId());


            assert(data_ptrs.size() == uids.size());
            int64_t num_ptrs = data_ptrs.size();
            std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
            auto variantPack = std::make_unique<cudnn_frontend::VariantPack>(cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                .setDataPointers(num_ptrs, data_ptrs.data())
                .setUids(num_ptrs, uids.data())
                .build());
            std::cout << "variantPack " << variantPack->describe() << std::endl;

            cudnnStatus_t status;
            status = cudnnBackendExecute(handle, plan->get_raw_desc(), variantPack->get_raw_desc());
            std::cout << cudnnGetErrorString(status) << std::endl;

            if (workspace_size > 0)
            {
                checkCudaErr(cudaFree(workspace_ptr));
            }
        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }

        cudaDeviceSynchronize();
        checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0])* Y.n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(Z.hostPtr, Z.devPtr, (size_t)(sizeof(Z.hostPtr[0])* Z.n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(E.hostPtr, E.devPtr, (size_t)(sizeof(E.hostPtr[0])* E.n_elems), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        display_flat(X.hostPtr, 0, 10);
        display_flat(Y.hostPtr, 0, 10);
        display_flat(Z.hostPtr, 0, 1);
        display_flat(E.hostPtr, 0, 1);


        if (handle)
        {
            cudnnDestroy(handle);
        }

    }

    void LenetForward_v_8()
    {
        std::cout << "Less rigid forward LeNet Test v0.8" << std::endl;
        
        // Hyperparameters?
        constexpr int64_t batchSize = 1;
        constexpr int64_t inputH = 28;
        constexpr int64_t inputW = 28;
        constexpr int64_t C1Features = 6;
        constexpr int64_t C1KernelSize = 5;
        constexpr int64_t C1Padding = 2;
        constexpr int64_t C3Features = 16;
        constexpr int64_t C3KernelSize = 5;
        constexpr int64_t C3Padding = 0;
        constexpr int64_t C5Features = 120;
        constexpr int64_t C5KernelSize = 5;
        constexpr int64_t C5Padding = 0;

        bool verbose = true;

        MNISTDataHolder dh;
        dh.initialize();
        auto [image, label] = dh.getNextTrain();
        auto [rows, cols] = dh.getDimensions();

        // Input
        int64_t xTensorDim[] = { batchSize, 1, inputH, inputW }; // input
        Helpers::Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3]);

        for (int64_t i = 0; i < X.n_elems; ++i)
        {
            X.hostPtr[i] = static_cast<float>(image[i]) / 256;
            std::cout << std::setw(3) << std::setprecision(3) << X.hostPtr[i] << " ";
            if (i > 1 && ((i + 1) % cols == 0))
            {
                std::cout << std::endl;
            }
        }

        checkCudaErr(cudaMemcpy(X.devPtr, X.hostPtr, size_t(sizeof(X.hostPtr[0]) * X.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle;
        checkCudnnErr(cudnnCreate(&handle));

        constexpr int64_t alignment = 16; //16
        const cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
        const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

        int64_t stride[4];
        Helpers::generateStrides(xTensorDim, stride, 4, tensorFormat);
        auto xTensor = cudnn_frontend::TensorBuilder()
            .setDim(4, xTensorDim)
            .setStride(4, stride)
            .setId('x')
            .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
            .setDataType(dataType)
            .build();

        // C1 - First convo

        ConvBiasAct C1(handle, xTensorDim, C1KernelSize, C1Features, xTensor, X.devPtr, C1Padding, verbose);
        Pool S2(handle, C1.getOutputTensor(), C1.getOutputSurface().devPtr, verbose);
        ConvBiasAct C3(handle, S2.getOutputTensor().getDim(), C3KernelSize, C3Features, S2.getOutputTensor(), S2.getOutputSurface().devPtr, C3Padding, verbose);
        Pool S4(handle, C3.getOutputTensor(), C3.getOutputSurface().devPtr, verbose);
        ConvBiasAct C5(handle, S4.getOutputTensor().getDim(), C5KernelSize, C5Features, S4.getOutputTensor(), S4.getOutputSurface().devPtr, C5Padding, verbose);
        //FC FC6(handle, C5.getOutputTensor(), C5.getOutputSurface().devPtr, 84, verbose);
        //FC FC7(handle, FC6.getOutputTensor(), FC6.getOutputSurface().devPtr, 10, verbose);
        //Softmax SM8(handle, FC7.getOutputTensor(), FC7.getOutputSurface(), true);

        C1.executeInference();
        S2.executeInference();
        C3.executeInference();
        S4.executeInference();
        C5.executeInference();
        //FC6.executeInference();
        //FC7.executeInference();
        //SM8.executeInference();

        C1.printOutput();
        S2.printOutput();
        C3.printOutput();
        S4.printOutput();
        C5.printOutput();
        //FC6.printOutput();
        //FC7.printOutput();
        //SM8.printOutput();

        if (handle) cudnnDestroy(handle);
    }

    void LenetForward_v_7()
    {
        std::cout << "Rigid LeNet Test v0.7" << std::endl;

        using namespace Helpers;

        MNISTDataHolder dh;
        dh.initialize();
        auto [image, label] = dh.getNextTrain();
        auto [rows, cols] = dh.getDimensions();

        //constexpr int64_t N = 1;
        //constexpr int64_t C = 1;
        constexpr int64_t inputH = 28;
        constexpr int64_t inputW = 28;
        constexpr int64_t C1Features = 6;
        constexpr int64_t C3Features = 16;

        // C1 - First convo
        int64_t xTensorDim[] = { 1, 1, 28, 28 }; // input
        int64_t wTensorDim[] = { C1Features, 1, 5, 5 }; // filter
        int64_t yTensorDim[] = { 0, 0, 0, 0 }; // Computed Below

        int64_t padA[] = { 2, 2 };
        int64_t dilationA[] = { 1, 1 };
        int64_t convstrideA[] = { 1, 1 };
        int64_t bTensorDim[] = { 1, wTensorDim[0], 1, 1 };  // bias

        yTensorDim[0] = xTensorDim[0];
        yTensorDim[1] = wTensorDim[0];
        for (int dim = 0; dim < 2; dim++) {
            yTensorDim[dim + 2] = Helpers::getFwdConvOutputDim(xTensorDim[dim + 2], padA[dim], wTensorDim[dim + 2], convstrideA[dim], dilationA[dim]);
        }

        printf("====DIMENSIONS====\n");
        printf("input dims are %lld, %lld, %lld, %lld\n", xTensorDim[0], xTensorDim[1], xTensorDim[2], xTensorDim[3]);
        printf("filter dims are %lld, %lld, %lld, %lld\n", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]);
        printf("output dims are %lld, %lld, %lld, %lld\n", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]);

        int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

        Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
        Surface<float> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
        Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
        Surface<float> Y(Ysize, false);

        int64_t* x_dim = xTensorDim;
        int64_t* w_dim = wTensorDim;
        int64_t* b_dim = bTensorDim;
        int64_t* y_dim = yTensorDim;

        void* devPtrX = X.devPtr;
        void* devPtrW = W.devPtr;
        void* devPtrY = Y.devPtr;
        void* devPtrB = B.devPtr;

        for (int64_t i = 0; i < X.n_elems; ++i)
        {
            X.hostPtr[i] = static_cast<float>(image[i]) / 256;
            std::cout << std::setw(3) << std::setprecision(3) << X.hostPtr[i] << " ";
            if (i > 1 && ((i + 1) % cols == 0))
            {
                std::cout << std::endl;
            }
        }

        for (int64_t i = 0; i < Y.n_elems; ++i)
        {
            Y.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(devPtrX, X.hostPtr, size_t(sizeof(X.hostPtr[0]) * X.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrY, Y.hostPtr, size_t(sizeof(Y.hostPtr[0]) * Y.n_elems), cudaMemcpyHostToDevice));

        // S2 - Pooling
        int64_t poolTensorDim[] = { 0, 0, 0, 0 };
        poolTensorDim[0] = yTensorDim[0];
        poolTensorDim[1] = yTensorDim[1];
        poolTensorDim[2] = yTensorDim[2] / 2;
        poolTensorDim[3] = yTensorDim[3] / 2;

        int64_t windowDimPool[CUDNN_DIM_MAX] = { 2,2 };
        int64_t prePaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        int64_t postPaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        int64_t stridePool[CUDNN_DIM_MAX] = { 2,2 };

        printf("After pool dims are %lld, %lld, %lld, %lld\n", poolTensorDim[0], poolTensorDim[1], poolTensorDim[2], poolTensorDim[3]);


        int64_t Psize = poolTensorDim[0] * poolTensorDim[1] * poolTensorDim[2] * poolTensorDim[3];
        Surface<float> P(Psize, false);

        int64_t* p_dim = poolTensorDim;
        void* devPtrP = P.devPtr;



        for (int64_t i = 0; i < P.n_elems; ++i)
        {
            P.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(devPtrP, P.hostPtr, size_t(sizeof(P.hostPtr[0]) * P.n_elems), cudaMemcpyHostToDevice));

        // C3 - Second convo
        int64_t C3InputDim[] = { poolTensorDim[0], poolTensorDim[1], poolTensorDim[2], poolTensorDim[3] };
        int64_t C3FilterDim[] = { C3Features, C3InputDim[1], 5, 5 }; // filter
        int64_t C3OutputDim[] = { 0, 0, 0, 0 }; // Computed Below

        int64_t C3Pad[] = { 0, 0 };
        int64_t C3Dilation[] = { 1, 1 };
        int64_t C3ConvStride[] = { 1, 1 };
        int64_t C3BiasDim[] = { 1, C3FilterDim[0], 1, 1 };  // bias

        C3OutputDim[0] = C3InputDim[0];
        C3OutputDim[1] = C3FilterDim[0];
        for (int dim = 0; dim < 2; dim++) {
            C3OutputDim[dim + 2] = Helpers::getFwdConvOutputDim(C3InputDim[dim + 2], C3Pad[dim], C3FilterDim[dim + 2], C3ConvStride[dim], C3Dilation[dim]);
        }

        printf("====DIMENSIONS====\n");
        printf("C3InputDim dims are %lld, %lld, %lld, %lld\n", C3InputDim[0], C3InputDim[1], C3InputDim[2], C3InputDim[3]);
        printf("C3filter dims are %lld, %lld, %lld, %lld\n", C3FilterDim[0], C3FilterDim[1], C3FilterDim[2], C3FilterDim[3]);
        printf("C3OutputDim dims are %lld, %lld, %lld, %lld\n", C3OutputDim[0], C3OutputDim[1], C3OutputDim[2], C3OutputDim[3]);

        int64_t C3Ysize = C3OutputDim[0] * C3OutputDim[1] * C3OutputDim[2] * C3OutputDim[3];

        Surface<float> C3X(C3InputDim[0] * C3InputDim[1] * C3InputDim[2] * C3InputDim[3], false);
        Surface<float> C3W(C3FilterDim[0] * C3FilterDim[1] * C3FilterDim[2] * C3FilterDim[3], false);
        Surface<float> C3B(C3BiasDim[0] * C3BiasDim[1] * C3BiasDim[2] * C3BiasDim[3], false);
        Surface<float> C3Y(C3Ysize, false);

        int64_t* C3x_dim = C3InputDim;
        int64_t* C3w_dim = C3FilterDim;
        int64_t* C3b_dim = C3BiasDim;
        int64_t* C3y_dim = C3OutputDim;

        void* C3devPtrX = C3X.devPtr;
        void* C3devPtrW = C3W.devPtr;
        void* C3devPtrY = C3Y.devPtr;
        void* C3devPtrB = C3B.devPtr;

        for (int64_t i = 0; i < C3Y.n_elems; ++i)
        {
            C3Y.hostPtr[i] = 1;
        }

        checkCudaErr(cudaMemcpy(C3devPtrY, C3Y.hostPtr, size_t(sizeof(C3Y.hostPtr[0]) * C3Y.n_elems), cudaMemcpyHostToDevice));


        // S4 - Pooling
        int64_t S4poolTensorDim[] = { 0, 0, 0, 0 };
        S4poolTensorDim[0] = C3OutputDim[0];
        S4poolTensorDim[1] = C3OutputDim[1];
        S4poolTensorDim[2] = C3OutputDim[2] / 2;
        S4poolTensorDim[3] = C3OutputDim[3] / 2;

        //int64_t windowDimPool[CUDNN_DIM_MAX] = { 2,2 };
        //int64_t prePaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        //int64_t postPaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        //int64_t stridePool[CUDNN_DIM_MAX] = { 2,2 };

        printf("After pool dims are %lld, %lld, %lld, %lld\n", S4poolTensorDim[0], S4poolTensorDim[1], S4poolTensorDim[2], S4poolTensorDim[3]);


        int64_t S4Psize = S4poolTensorDim[0] * S4poolTensorDim[1] * S4poolTensorDim[2] * S4poolTensorDim[3];
        Surface<float> S4P(S4Psize, false);

        int64_t* S4p_dim = S4poolTensorDim;
        void* S4devPtrP = S4P.devPtr;

        for (int64_t i = 0; i < S4P.n_elems; ++i)
        {
            S4P.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(S4devPtrP, S4P.hostPtr, size_t(sizeof(S4P.hostPtr[0]) * S4P.n_elems), cudaMemcpyHostToDevice));


        // FC5 - Fully connected layer done with MatMul
        constexpr int64_t FC5NumOutput = 120; //hyperparameter
        int64_t flattenS4NumOutput = S4poolTensorDim[1] * S4poolTensorDim[2] * S4poolTensorDim[3];

        int64_t FC5InputTensorDim[] = { 1, 1, flattenS4NumOutput }; //batch M K
        int64_t FC5WeightsTensorDim[] = { 1, FC5InputTensorDim[2], FC5NumOutput}; //batch K N
        int64_t FC5OutputTensorDim[] = { 1, FC5InputTensorDim[1], FC5WeightsTensorDim[2]}; //batch M N

        int64_t FC5BiasTensorDim[] = { 1, 1, FC5NumOutput };  //bias


        printf("====DIMENSIONS====\n");
        std::cout << std::format("a matrix dims are {}, {}, {}", FC5InputTensorDim[0], FC5InputTensorDim[1], FC5InputTensorDim[2]) << std::endl;
        std::cout << std::format("b matrix dims are {}, {}, {}", FC5WeightsTensorDim[0], FC5WeightsTensorDim[1], FC5WeightsTensorDim[2]) << std::endl;
        std::cout << std::format("c matrix dims are {}, {}, {}", FC5OutputTensorDim[0], FC5OutputTensorDim[1], FC5OutputTensorDim[2]) << std::endl;

        int64_t FC5Csize = FC5OutputTensorDim[0] * FC5OutputTensorDim[1] * FC5OutputTensorDim[2];

        //Surface<float> FC5A(FC5InputTensorDim[0] * FC5InputTensorDim[1] * FC5InputTensorDim[2], false);
        Surface<float> FC5B(FC5WeightsTensorDim[0] * FC5WeightsTensorDim[1] * FC5WeightsTensorDim[2], false);
        Surface<float> FC5C(FC5Csize, true);

        Surface<float> FC5Z(FC5BiasTensorDim[0] * FC5BiasTensorDim[1] * FC5BiasTensorDim[2], false);
        Surface<float> FC5AfterZ(FC5Csize, false);

        int64_t* FC5a_dim = FC5InputTensorDim;
        int64_t* FC5b_dim = FC5WeightsTensorDim;
        int64_t* FC5c_dim = FC5OutputTensorDim;
        int64_t* FC5z_dim = FC5BiasTensorDim;
        //void* FC5devPtrA = FC5A.devPtr;
        void* FC5devPtrB = FC5B.devPtr;
        void* FC5devPtrC = FC5C.devPtr;
        void* FC5devPtrZ = FC5Z.devPtr;
        void* FC5devPtrAfterZ = FC5AfterZ.devPtr;

        for (int64_t i = 0; i < FC5C.n_elems; ++i)
        {
            FC5C.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(FC5devPtrC, FC5C.hostPtr, size_t(sizeof(FC5C.hostPtr[0]) * FC5C.n_elems), cudaMemcpyHostToDevice));


        // FC6 - Fully connected layer done with MatMul
        constexpr int64_t FC6NumOutput = 84; //hyperparameter
        int64_t flattenFC5NumOutput = FC5OutputTensorDim[1] * FC5OutputTensorDim[2];

        int64_t FC6InputTensorDim[] = { 1, 1, flattenFC5NumOutput }; //batch M K
        int64_t FC6WeightsTensorDim[] = { 1, FC6InputTensorDim[2], FC6NumOutput }; //batch K N
        int64_t FC6OutputTensorDim[] = { 1, FC6InputTensorDim[1], FC6WeightsTensorDim[2] }; //batch M N

        int64_t FC6BiasTensorDim[] = { 1, 1, FC6NumOutput };  //bias


        printf("====DIMENSIONS====\n");
        std::cout << std::format("a matrix dims are {}, {}, {}", FC6InputTensorDim[0], FC6InputTensorDim[1], FC6InputTensorDim[2]) << std::endl;
        std::cout << std::format("b matrix dims are {}, {}, {}", FC6WeightsTensorDim[0], FC6WeightsTensorDim[1], FC6WeightsTensorDim[2]) << std::endl;
        std::cout << std::format("c matrix dims are {}, {}, {}", FC6OutputTensorDim[0], FC6OutputTensorDim[1], FC6OutputTensorDim[2]) << std::endl;

        int64_t FC6Csize = FC6OutputTensorDim[0] * FC6OutputTensorDim[1] * FC6OutputTensorDim[2];

        //Surface<float> FC6A(FC6InputTensorDim[0] * FC6InputTensorDim[1] * FC6InputTensorDim[2], false);
        Surface<float> FC6B(FC6WeightsTensorDim[0] * FC6WeightsTensorDim[1] * FC6WeightsTensorDim[2], false);
        Surface<float> FC6C(FC6Csize, true);

        Surface<float> FC6Z(FC6BiasTensorDim[0] * FC6BiasTensorDim[1] * FC6BiasTensorDim[2], false);
        Surface<float> FC6AfterZ(FC6Csize, false);

        int64_t* FC6a_dim = FC6InputTensorDim;
        int64_t* FC6b_dim = FC6WeightsTensorDim;
        int64_t* FC6c_dim = FC6OutputTensorDim;
        int64_t* FC6z_dim = FC6BiasTensorDim;
        //void* FC6devPtrA = FC6A.devPtr;
        void* FC6devPtrB = FC6B.devPtr;
        void* FC6devPtrC = FC6C.devPtr;
        void* FC6devPtrZ = FC6Z.devPtr;
        void* FC6devPtrAfterZ = FC6AfterZ.devPtr;

        for (int64_t i = 0; i < FC6C.n_elems; ++i)
        {
            FC6C.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(FC6devPtrC, FC6C.hostPtr, size_t(sizeof(FC6C.hostPtr[0]) * FC6C.n_elems), cudaMemcpyHostToDevice));


        // FC7 - Fully connected layer to output
        constexpr int64_t FC7NumOutput = 10; //hyperparameter - actual number of classes
        int64_t flattenFC7NumOutput = FC6OutputTensorDim[1] * FC6OutputTensorDim[2];

        int64_t FC7InputTensorDim[] = { 1, 1, flattenFC7NumOutput }; //batch M K
        int64_t FC7WeightsTensorDim[] = { 1, FC7InputTensorDim[2], FC7NumOutput }; //batch K N
        int64_t FC7OutputTensorDim[] = { 1, FC7InputTensorDim[1], FC7WeightsTensorDim[2] }; //batch M N

        int64_t FC7BiasTensorDim[] = { 1, 1, FC7NumOutput };  //bias


        printf("====DIMENSIONS====\n");
        std::cout << std::format("a matrix dims are {}, {}, {}", FC7InputTensorDim[0], FC7InputTensorDim[1], FC7InputTensorDim[2]) << std::endl;
        std::cout << std::format("b matrix dims are {}, {}, {}", FC7WeightsTensorDim[0], FC7WeightsTensorDim[1], FC7WeightsTensorDim[2]) << std::endl;
        std::cout << std::format("c matrix dims are {}, {}, {}", FC7OutputTensorDim[0], FC7OutputTensorDim[1], FC7OutputTensorDim[2]) << std::endl;

        int64_t FC7Csize = FC7OutputTensorDim[0] * FC7OutputTensorDim[1] * FC7OutputTensorDim[2];

        //Surface<float> FC7A(FC7InputTensorDim[0] * FC7InputTensorDim[1] * FC7InputTensorDim[2], false);
        Surface<float> FC7B(FC7WeightsTensorDim[0] * FC7WeightsTensorDim[1] * FC7WeightsTensorDim[2], false);
        Surface<float> FC7C(FC7Csize, true);

        Surface<float> FC7Z(FC7BiasTensorDim[0] * FC7BiasTensorDim[1] * FC7BiasTensorDim[2], false);
        Surface<float> FC7AfterZ(FC7Csize, false);

        int64_t* FC7a_dim = FC7InputTensorDim;
        int64_t* FC7b_dim = FC7WeightsTensorDim;
        int64_t* FC7c_dim = FC7OutputTensorDim;
        int64_t* FC7z_dim = FC7BiasTensorDim;
        //void* FC7devPtrA = FC7A.devPtr;
        void* FC7devPtrB = FC7B.devPtr;
        void* FC7devPtrC = FC7C.devPtr;
        void* FC7devPtrZ = FC7Z.devPtr;
        void* FC7devPtrAfterZ = FC7AfterZ.devPtr;

        for (int64_t i = 0; i < FC7C.n_elems; ++i)
        {
            FC7C.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(FC7devPtrC, FC7C.hostPtr, size_t(sizeof(FC7C.hostPtr[0]) * FC7C.n_elems), cudaMemcpyHostToDevice));


        // Sofmax
        int64_t softmaxTensorDim[] = { 1, 1, FC7NumOutput };
        Surface<float> S(FC7NumOutput);

        for (int64_t i = 0; i < S.n_elems; ++i)
        {
            S.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(S.devPtr, S.hostPtr, size_t(sizeof(S.hostPtr[0]) * S.n_elems), cudaMemcpyHostToDevice));


        checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle_;

        try
        {
            checkCudnnErr(cudnnCreate(&handle_));

            constexpr int64_t alignment = 16; //16
            cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
            cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
            constexpr int convDim = 2;
            float alpha = 1.0f;
            float beta = 0.0f;
            //cudnn_frontend::ExecutionPlan plan;

            // C1 - convolution
            int64_t stride[4];
            generateStrides(x_dim, stride, 4, tensorFormat);
            auto xTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, x_dim)
                .setStride(4, stride)
                .setId('x')
                .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();
            generateStrides(w_dim, stride, 4, tensorFormat);
            auto wTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, w_dim)
                .setStride(4, stride)
                .setId('w')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(b_dim, stride, 4, tensorFormat);
            auto bTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, b_dim)
                .setStride(4, stride)
                .setId('b')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(y_dim, stride, 4, tensorFormat);
            auto afterConvTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('A')  // after conv
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('B')  // after bias
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto yTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('y')  // after relu
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            std::cout << xTensor.describe() << std::endl;
            std::cout << wTensor.describe() << std::endl;
            std::cout << afterConvTensor.describe() << std::endl;
            std::cout << yTensor.describe() << std::endl;

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
            std::cout << convDesc.describe() << std::endl;

            // Define the bias descriptor
            auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_RELU_FWD)//CUDNN_POINTWISE_SIGMOID_FWD
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << actDesc.describe() << std::endl;

            // Create a convolution Node
            auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(xTensor)
                .setwDesc(wTensor)
                .setyDesc(afterConvTensor) //afterConvTensor // yTensor
                .setcDesc(convDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << conv_op.describe() << std::endl;

            // Create a Bias Node.
            auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(conv_op.getOutputTensor()) //conv_op.getOutputTensor()
                .setbDesc(bTensor)
                .setyDesc(afterBiasTensor) //afterBiasTensor
                .setpwDesc(biasDesc)
                .build();
            std::cout << bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(bias_op.getOutputTensor()) // bias_op.getOutputTensor()
                .setyDesc(yTensor)
                .setpwDesc(actDesc)
                .build();
            std::cout << act_op.describe() << std::endl;

            // Create an Operation Graph.
            //std::vector<cudnn_frontend::Operation const*> ops = { &conv_op,  &bias_op, &act_op, &pool_op};
            std::vector<cudnn_frontend::Operation const*> ops;
            ops.emplace_back(&conv_op);
            ops.emplace_back(&bias_op);
            ops.emplace_back(&act_op);

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            std::cout << opGraph.describe() << std::endl;

            auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

            std::cout << "Plan tag: " << plan.getTag() << std::endl;

            int64_t workspace_size = 0;
            void* workspace_ptr = nullptr;

            workspace_size = plan.getWorkspaceSize();
            std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }
            //void* data_ptrs[] = { devPtrX, devPtrW, devPtrY, devPtrB, devPtrP };
            //int64_t uids[] = { 'x', 'w', 'y', 'b', 'p'};
            std::vector<void*> data_ptrs;
            data_ptrs.emplace_back(devPtrX);
            data_ptrs.emplace_back(devPtrW);
            data_ptrs.emplace_back(devPtrB);
            data_ptrs.emplace_back(devPtrY);

            std::vector<int64_t> uids;
            uids.emplace_back('x');
            uids.emplace_back('w');
            uids.emplace_back('b');
            uids.emplace_back('y');

            assert(data_ptrs.size() == uids.size());
            //int64_t num_ptrs = sizeof(uids) / sizeof(uids[0]);
            int64_t num_ptrs = data_ptrs.size();
            std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
            auto variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                //.setDataPointers(num_ptrs, data_ptrs)
                .setDataPointers(num_ptrs, data_ptrs.data())
                .setUids(num_ptrs, uids.data())
                .build();
            std::cout << "variantPack " << variantPack.describe() << std::endl;


            // S2 - Pooling

            auto const nanOpt = CUDNN_NOT_PROPAGATE_NAN;
            constexpr int64_t nbSpatialDims = 2;
            cudnn_frontend::cudnnResampleMode_t const mode = cudnn_frontend::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
            cudnn_frontend::cudnnPaddingMode_t const padding_mode = cudnn_frontend::cudnnPaddingMode_t::CUDNN_ZERO_PAD;

            generateStrides(p_dim, stride, 4, tensorFormat);
            auto afterPoolTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, p_dim)
                .setStride(4, stride)
                .setId('p')  // after pool
                .setAlignment(16)
                .setDataType(dataType)
                .build();
            std::cout << afterPoolTensor.describe() << std::endl;

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
            std::cout << "Initialized Pool Desc" << std::endl;
            std::cout << poolDesc.describe() << std::endl;

            // Create a Resample Node
            auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                .setxDesc(act_op.getOutputTensor())
                .setyDesc(afterPoolTensor)
                .setResampleDesc(poolDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << pool_op.describe() << std::endl;

            std::vector< cudnn_frontend::Operation const*> poolOps = { &pool_op };

            auto poolOpGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(poolOps.size(), poolOps.data())
                .build();

            std::cout << poolOpGraph.describe() << std::endl;

            auto poolPlan = get_execplan_from_heuristics_else_fall_back(std::move(poolOpGraph), handle_);

            std::cout << "poolPlan tag: " << poolPlan.getTag() << std::endl;

            int64_t workspace_size_pool = 0;
            void* workspace_ptr_pool = nullptr;

            workspace_size_pool = poolPlan.getWorkspaceSize();
            std::cout << poolPlan.describe() << " requires workspace " << workspace_size_pool << std::endl;

            if (workspace_size_pool > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr_pool, (size_t)workspace_size_pool));
            }

            std::vector<void*> data_ptrs_pool;
            data_ptrs_pool.emplace_back(devPtrY);
            data_ptrs_pool.emplace_back(devPtrP);

            std::vector<int64_t> uids_pool;
            uids_pool.emplace_back('y');
            uids_pool.emplace_back('p');

            assert(data_ptrs_pool.size() == uids_pool.size());
            int64_t num_ptrs_pool = data_ptrs_pool.size();
            std::cout << std::format("Num ptrs {}", num_ptrs_pool) << std::endl;
            auto variantPackPool = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr_pool)
                .setDataPointers(num_ptrs_pool, data_ptrs_pool.data())
                .setUids(num_ptrs_pool, uids_pool.data())
                .build();
            std::cout << "variantPack " << variantPackPool.describe() << std::endl;


            // C3 - convolution
            int64_t id = 0;
            generateStrides(C3x_dim, stride, 4, tensorFormat);
            auto C3xTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3x_dim)
                .setStride(4, stride)
                .setId(id++) // 0
                .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();

            generateStrides(C3w_dim, stride, 4, tensorFormat);
            auto C3wTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3w_dim)
                .setStride(4, stride)
                .setId(id++) // 1
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(C3b_dim, stride, 4, tensorFormat);
            auto C3bTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3b_dim)
                .setStride(4, stride)
                .setId(id++) // 2
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(C3y_dim, stride, 4, tensorFormat);
            auto C3yTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3y_dim)
                .setStride(4, stride)
                .setId(id++)  // after relu // 3
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            auto C3afterConvTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3y_dim)
                .setStride(4, stride)
                .setId(id++)  // after conv // 4
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto C3afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3y_dim)
                .setStride(4, stride)
                .setId(id++)  // after bias // 5
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            std::cout << C3xTensor.describe() << std::endl;
            std::cout << C3wTensor.describe() << std::endl;
            std::cout << C3bTensor.describe() << std::endl;
            std::cout << C3yTensor.describe() << std::endl;
            std::cout << C3afterConvTensor.describe() << std::endl;
            std::cout << C3afterBiasTensor.describe() << std::endl;

            // Define the convolution problem
            auto C3convDesc = cudnn_frontend::ConvDescBuilder()
                .setComputeType(CUDNN_DATA_FLOAT)
                .setMathMode(CUDNN_CROSS_CORRELATION)
                .setSpatialDimCount(convDim)
                .setSpatialStride(convDim, C3ConvStride)
                .setPrePadding(convDim, C3Pad)
                .setPostPadding(convDim, C3Pad)
                .setDilation(convDim, C3Dilation)
                .build();
            std::cout << C3convDesc.describe() << std::endl;

            // Define the bias descriptor
            auto C3biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << C3biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto C3actDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_RELU_FWD)//CUDNN_POINTWISE_SIGMOID_FWD
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << C3actDesc.describe() << std::endl;

            // Create a convolution Node
            auto C3conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(pool_op.getOutputTensor()) //pool_op.getOutputTensor()
                .setwDesc(C3wTensor)
                .setyDesc(C3afterConvTensor) //C3afterConvTensor // C3yTensor
                .setcDesc(C3convDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << C3conv_op.describe() << std::endl;

            // Create a Bias Node.
            auto C3bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(C3conv_op.getOutputTensor())
                .setbDesc(C3bTensor)
                .setyDesc(C3afterBiasTensor) //afterBiasTensor
                .setpwDesc(C3biasDesc)
                .build();
            std::cout << C3bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto C3act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(C3bias_op.getOutputTensor()) // bias_op.getOutputTensor()
                .setyDesc(C3yTensor)
                .setpwDesc(C3actDesc)
                .build();
            std::cout << C3act_op.describe() << std::endl;

            // Create an Operation Graph.
            std::vector<cudnn_frontend::Operation const*> C3ops;
            C3ops.emplace_back(&C3conv_op);
            C3ops.emplace_back(&C3bias_op);
            C3ops.emplace_back(&C3act_op);

            auto C3opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(C3ops.size(), C3ops.data())
                .build();

            std::cout << C3opGraph.describe() << std::endl;

            auto C3plan = get_execplan_from_heuristics_else_fall_back(std::move(C3opGraph), handle_);

            std::cout << "C3Plan tag: " << C3plan.getTag() << std::endl;

            auto C3workspace_size = C3plan.getWorkspaceSize();
            std::cout << C3plan.describe() << " requires workspace " << C3workspace_size << std::endl;

            void* C3workspace_ptr = nullptr;
            if (C3workspace_size > 0) {
                checkCudaErr(cudaMalloc(&C3workspace_ptr, (size_t)C3workspace_size));
            }
            std::vector<void*> C3data_ptrs;
            //C3data_ptrs.emplace_back(C3devPtrX);
            C3data_ptrs.emplace_back(devPtrP);
            C3data_ptrs.emplace_back(C3devPtrW);
            C3data_ptrs.emplace_back(C3devPtrB);
            C3data_ptrs.emplace_back(C3devPtrY);

            std::vector<int64_t> C3uids;
            //C3uids.emplace_back(0);
            C3uids.emplace_back('p');
            C3uids.emplace_back(1);
            C3uids.emplace_back(2);
            C3uids.emplace_back(3);

            assert(C3data_ptrs.size() == C3uids.size());
            int64_t C3num_ptrs = C3data_ptrs.size();
            std::cout << std::format("Num ptrs {}", C3num_ptrs) << std::endl;
            auto C3variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(C3workspace_ptr)
                .setDataPointers(C3num_ptrs, C3data_ptrs.data())
                .setUids(C3num_ptrs, C3uids.data())
                .build();
            std::cout << "variantPack " << C3variantPack.describe() << std::endl;


            // S4 - Pooling
            generateStrides(S4p_dim, stride, 4, tensorFormat);
            auto S4afterPoolTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, S4p_dim)
                .setStride(4, stride)
                .setId(id++)  // after pool
                .setAlignment(16)
                .setDataType(dataType)
                .build();
            std::cout << afterPoolTensor.describe() << std::endl;

            // Define the resample descriptor
            auto S4poolDesc = cudnn_frontend::ResampleDescBuilder()
                .setComputeType(CUDNN_DATA_FLOAT)
                .setNanPropagation(nanOpt)
                .setResampleMode(mode)
                .setPaddingMode(padding_mode)
                .setSpatialDim(nbSpatialDims, windowDimPool)
                .setSpatialStride(nbSpatialDims, stridePool)
                .setPrePadding(nbSpatialDims, prePaddingPool)
                .setPostPadding(nbSpatialDims, postPaddingPool)
                .build();
            std::cout << "Initialized Pool Desc" << std::endl;
            std::cout << S4poolDesc.describe() << std::endl;

            // Create a Resample Node
            auto S4pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                .setxDesc(C3act_op.getOutputTensor())
                .setyDesc(S4afterPoolTensor)
                .setResampleDesc(S4poolDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << S4pool_op.describe() << std::endl;

            std::vector< cudnn_frontend::Operation const*> S4poolOps = { &S4pool_op };

            auto S4poolOpGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(S4poolOps.size(), S4poolOps.data())
                .build();

            std::cout << S4poolOpGraph.describe() << std::endl;

            auto S4poolPlan = get_execplan_from_heuristics_else_fall_back(std::move(S4poolOpGraph), handle_);

            std::cout << "poolPlan tag: " << S4poolPlan.getTag() << std::endl;

            int64_t S4workspace_size_pool = 0;
            void* S4workspace_ptr_pool = nullptr;

            S4workspace_size_pool = poolPlan.getWorkspaceSize();
            std::cout << S4poolPlan.describe() << " requires workspace " << S4workspace_size_pool << std::endl;

            if (S4workspace_size_pool > 0) {
                checkCudaErr(cudaMalloc(&S4workspace_ptr_pool, (size_t)S4workspace_size_pool));
            }

            std::vector<void*> S4data_ptrs_pool;
            S4data_ptrs_pool.emplace_back(C3devPtrY);
            S4data_ptrs_pool.emplace_back(S4devPtrP);

            std::vector<int64_t> S4uids_pool;
            S4uids_pool.emplace_back(C3yTensor.getId());
            S4uids_pool.emplace_back(S4afterPoolTensor.getId());

            assert(S4data_ptrs_pool.size() == S4uids_pool.size());
            int64_t S4num_ptrs_pool = S4data_ptrs_pool.size();
            std::cout << std::format("Num ptrs {}", S4num_ptrs_pool) << std::endl;
            auto S4variantPackPool = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(S4workspace_ptr_pool)
                .setDataPointers(S4num_ptrs_pool, S4data_ptrs_pool.data())
                .setUids(S4num_ptrs_pool, S4uids_pool.data())
                .build();
            std::cout << "variantPack " << S4variantPackPool.describe() << std::endl;


            // FC5 - MatMul
            //if (check_device_arch_newer_than("ampere") == false && dataType == CUDNN_DATA_FLOAT) {
            //    cudnn_frontend::set_error_and_throw_exception(
            //        nullptr,
            //        CUDNN_STATUS_ARCH_MISMATCH,
            //        "run_matmul_bias_gelu: Sample requires Ampere or above GPU");
            //}
            // Creates the necessary tensor descriptors
            int64_t FCstride[3];
            // the intension is to compute stride for a [1, M, K] matrix with K in the inner most dimension, and
            // CUDNN_TENSOR_NCHW is a borrowed notation
            generateStrides(FC5a_dim, FCstride, 3, tensorFormat);
            auto FC5aMatrixTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC5a_dim)
                .setStride(3, FCstride)
                .setId('a')
                .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();
            generateStrides(FC5b_dim, FCstride, 3, tensorFormat);
            auto FC5bMatrixTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC5b_dim)
                .setStride(3, FCstride)
                .setId('b')
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            generateStrides(FC5z_dim, FCstride, 3, tensorFormat);
            auto FC5biasTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC5z_dim)
                .setStride(3, FCstride)
                .setId('z')
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            generateStrides(FC5c_dim, FCstride, 3, tensorFormat);
            auto FC5afterMatMulTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC5c_dim)
                .setStride(3, FCstride)
                .setId('A')  // after matmul
                .setAlignment(16)
                .setVirtual()
                .setDataType(dataType)
                .build();
            auto FC5afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC5c_dim)
                .setStride(3, FCstride)
                .setId('B')  // after bias
                .setAlignment(16)
                .setDataType(dataType)
                .build();
            auto FC5outputTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC5c_dim)
                .setStride(3, FCstride)
                .setId('c')  // output after gelu
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            std::cout << FC5aMatrixTensor.describe() << std::endl;
            std::cout << FC5bMatrixTensor.describe() << std::endl;
            std::cout << FC5biasTensor.describe() << std::endl;
            std::cout << FC5afterMatMulTensor.describe() << std::endl;
            std::cout << FC5afterBiasTensor.describe() << std::endl;
            std::cout << FC5outputTensor.describe() << std::endl;

            // Define the bias descriptor
            auto FC5biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << FC5biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto FC5actDesc = cudnn_frontend::PointWiseDescBuilder()
#if (CUDNN_VERSION >= 8500)
                .setMode(CUDNN_POINTWISE_GELU_APPROX_TANH_FWD)
#else
                .setMode(CUDNN_POINTWISE_GELU_FWD)
#endif
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << actDesc.describe() << std::endl;

            // Define the matmul desc
            auto FC5matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
            std::cout << FC5matmulDesc.describe() << std::endl;

            // Create a matmul Node
            auto FC5matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                .setaMatDesc(FC5aMatrixTensor)
                .setbMatDesc(FC5bMatrixTensor)
                .setcMatDesc(FC5afterMatMulTensor)
                .setmatmulDesc(FC5matmulDesc)
                .build();
            std::cout << FC5matmul_op.describe() << std::endl;

            // Create a Bias Node.
            auto FC5bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(FC5matmul_op.getOutputTensor())
                .setbDesc(FC5biasTensor)
                .setyDesc(FC5afterBiasTensor)
                .setpwDesc(FC5biasDesc)
                .build();
            std::cout << FC5bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto FC5act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(FC5bias_op.getOutputTensor())
                .setyDesc(FC5outputTensor)
                .setpwDesc(FC5actDesc)
                .build();
            std::cout << FC5act_op.describe() << std::endl;

            // Create an Operation Graph. In this case it is matmul bias activation
            std::array<cudnn_frontend::Operation const*, 3> FC5ops = { &FC5matmul_op, &FC5bias_op, &FC5act_op };

            auto FC5opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(FC5ops.size(), FC5ops.data())
                .build();

            auto FC5plan = get_execplan_from_heuristics_else_fall_back(std::move(FC5opGraph), handle_);

            std::cout << "Plan tag: " << FC5plan.getTag() << std::endl;

            auto FC5workspace_size = FC5plan.getWorkspaceSize();
            std::cout << FC5plan.describe() << " requires workspace " << FC5workspace_size << std::endl;

            void* FC5workspace_ptr = nullptr;
            if (FC5workspace_size > 0) {
                checkCudaErr(cudaMalloc(&FC5workspace_ptr, (size_t)FC5workspace_size));
            }
            void* FC5data_ptrs[] = { /*FC5devPtrA*/ S4devPtrP, FC5devPtrB, FC5devPtrC, FC5devPtrZ, FC5devPtrAfterZ };
            int64_t FC5uids[] = { 'a', 'b', 'c', 'z', 'B' };
            auto FC5variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(FC5workspace_ptr)
                .setDataPointers(5, FC5data_ptrs)
                .setUids(5, FC5uids)
                .build();
            std::cout << "variantPack " << FC5variantPack.describe() << std::endl;



            // FC6 - MatMul
            //generateStrides(FC6a_dim, FCstride, 3, tensorFormat);
            //auto FC6aMatrixTensor = cudnn_frontend::TensorBuilder()
            //    .setDim(3, FC6a_dim)
            //    .setStride(3, FCstride)
            //    .setId('a')
            //    .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
            //    .setDataType(dataType)
            //    .build();
            generateStrides(FC6b_dim, FCstride, 3, tensorFormat);
            auto FC6bMatrixTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC6b_dim)
                .setStride(3, FCstride)
                .setId('b')
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            generateStrides(FC6z_dim, FCstride, 3, tensorFormat);
            auto FC6biasTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC6z_dim)
                .setStride(3, FCstride)
                .setId('z')
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            generateStrides(FC6c_dim, FCstride, 3, tensorFormat);
            auto FC6afterMatMulTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC6c_dim)
                .setStride(3, FCstride)
                .setId('A')  // after matmul
                .setAlignment(16)
                .setVirtual()
                .setDataType(dataType)
                .build();
            auto FC6afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC6c_dim)
                .setStride(3, FCstride)
                .setId('B')  // after bias
                .setAlignment(16)
                .setDataType(dataType)
                .build();
            auto FC6outputTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC6c_dim)
                .setStride(3, FCstride)
                .setId('F')  // output after gelu
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            //std::cout << FC6aMatrixTensor.describe() << std::endl;
            std::cout << FC6bMatrixTensor.describe() << std::endl;
            std::cout << FC6biasTensor.describe() << std::endl;
            std::cout << FC6afterMatMulTensor.describe() << std::endl;
            std::cout << FC6afterBiasTensor.describe() << std::endl;
            std::cout << FC6outputTensor.describe() << std::endl;

            // Define the bias descriptor
            auto FC6biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << FC6biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto FC6actDesc = cudnn_frontend::PointWiseDescBuilder()
#if (CUDNN_VERSION >= 8500)
                .setMode(CUDNN_POINTWISE_GELU_APPROX_TANH_FWD)
#else
                .setMode(CUDNN_POINTWISE_GELU_FWD)
#endif
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << actDesc.describe() << std::endl;

            // Define the matmul desc
            auto FC6matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
            std::cout << FC6matmulDesc.describe() << std::endl;

            // Create a matmul Node
            auto FC6matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                .setaMatDesc(FC5outputTensor)
                .setbMatDesc(FC6bMatrixTensor)
                .setcMatDesc(FC6afterMatMulTensor)
                .setmatmulDesc(FC6matmulDesc)
                .build();
            std::cout << FC6matmul_op.describe() << std::endl;

            // Create a Bias Node.
            auto FC6bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(FC6matmul_op.getOutputTensor())
                .setbDesc(FC6biasTensor)
                .setyDesc(FC6afterBiasTensor)
                .setpwDesc(FC6biasDesc)
                .build();
            std::cout << FC6bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto FC6act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(FC6bias_op.getOutputTensor())
                .setyDesc(FC6outputTensor)
                .setpwDesc(FC6actDesc)
                .build();
            std::cout << FC6act_op.describe() << std::endl;

            // Create an Operation Graph. In this case it is matmul bias activation
            std::array<cudnn_frontend::Operation const*, 3> FC6ops = { &FC6matmul_op, &FC6bias_op, &FC6act_op };

            auto FC6opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(FC6ops.size(), FC6ops.data())
                .build();

            auto FC6plan = get_execplan_from_heuristics_else_fall_back(std::move(FC6opGraph), handle_);

            std::cout << "Plan tag: " << FC6plan.getTag() << std::endl;

            auto FC6workspace_size = FC6plan.getWorkspaceSize();
            std::cout << FC6plan.describe() << " requires workspace " << FC6workspace_size << std::endl;

            void* FC6workspace_ptr = nullptr;
            if (FC6workspace_size > 0) {
                checkCudaErr(cudaMalloc(&FC6workspace_ptr, (size_t)FC6workspace_size));
            }
            void* FC6data_ptrs[] = { /*FC6devPtrA*/ FC5devPtrC, FC6devPtrB, FC6devPtrC, FC6devPtrZ, FC6devPtrAfterZ };
            int64_t FC6uids[] = { 'c', 'b', 'F', 'z', 'B' };
            auto FC6variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(FC6workspace_ptr)
                .setDataPointers(5, FC6data_ptrs)
                .setUids(5, FC6uids)
                .build();
            std::cout << "variantPack " << FC6variantPack.describe() << std::endl;



            // FC7 - MatMul to output
            generateStrides(FC7b_dim, FCstride, 3, tensorFormat);
            auto FC7bMatrixTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC7b_dim)
                .setStride(3, FCstride)
                .setId('b')
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            generateStrides(FC7z_dim, FCstride, 3, tensorFormat);
            auto FC7biasTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC7z_dim)
                .setStride(3, FCstride)
                .setId('z')
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            generateStrides(FC7c_dim, FCstride, 3, tensorFormat);
            auto FC7afterMatMulTensor = cudnn_frontend::TensorBuilder()
                .setDim(3, FC7c_dim)
                .setStride(3, FCstride)
                .setId('A')  // after matmul
                .setAlignment(16)
                .setVirtual()
                .setDataType(dataType)
                .build();
            auto FC7outputTensor = cudnn_frontend::TensorBuilder() //FC7afterBiasTensor
                .setDim(3, FC7c_dim)
                .setStride(3, FCstride)
                .setId('O')  // after bias
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            //std::cout << FC7aMatrixTensor.describe() << std::endl;
            std::cout << FC7bMatrixTensor.describe() << std::endl;
            std::cout << FC7biasTensor.describe() << std::endl;
            std::cout << FC7afterMatMulTensor.describe() << std::endl;
            //std::cout << FC7afterBiasTensor.describe() << std::endl;
            std::cout << FC7outputTensor.describe() << std::endl;

            // Define the bias descriptor
            auto FC7biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << FC7biasDesc.describe() << std::endl;

            // Define the matmul desc
            auto FC7matmulDesc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
            std::cout << FC7matmulDesc.describe() << std::endl;

            // Create a matmul Node
            auto FC7matmul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                .setaMatDesc(FC6outputTensor)
                .setbMatDesc(FC7bMatrixTensor)
                .setcMatDesc(FC7afterMatMulTensor)
                .setmatmulDesc(FC7matmulDesc)
                .build();
            std::cout << FC7matmul_op.describe() << std::endl;

            // Create a Bias Node.
            auto FC7bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(FC7matmul_op.getOutputTensor())
                .setbDesc(FC7biasTensor)
                .setyDesc(FC7outputTensor)
                .setpwDesc(FC7biasDesc)
                .build();
            std::cout << FC7bias_op.describe() << std::endl;

            // Create an Operation Graph. In this case it is matmul bias activation
            std::array<cudnn_frontend::Operation const*, 2> FC7ops = { &FC7matmul_op, &FC7bias_op };

            auto FC7opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(FC7ops.size(), FC7ops.data())
                .build();

            auto FC7plan = get_execplan_from_heuristics_else_fall_back(std::move(FC7opGraph), handle_);

            std::cout << "Plan tag: " << FC7plan.getTag() << std::endl;

            auto FC7workspace_size = FC7plan.getWorkspaceSize();
            std::cout << FC7plan.describe() << " requires workspace " << FC7workspace_size << std::endl;

            void* FC7workspace_ptr = nullptr;
            if (FC7workspace_size > 0) {
                checkCudaErr(cudaMalloc(&FC7workspace_ptr, (size_t)FC7workspace_size));
            }
            void* FC7data_ptrs[] = {FC6devPtrC, FC7devPtrB, FC7devPtrC, FC7devPtrZ/*, FC7devPtrAfterZ */};
            int64_t FC7uids[] = { 'F', 'b', 'O', 'z'/*, 'B'*/ };
            auto FC7variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(FC7workspace_ptr)
                .setDataPointers(4, FC7data_ptrs)
                .setUids(4, FC7uids)
                .build();
            std::cout << "variantPack " << FC7variantPack.describe() << std::endl;


            //// Sofmax
            //generateStrides(softmaxTensorDim, FCstride, 3, tensorFormat);
            //auto softmaxTensor = cudnn_frontend::TensorBuilder()
            //    .setDim(3, softmaxTensorDim)
            //    .setStride(3, FCstride)
            //    .setId('R')
            //    .setAlignment(16)
            //    .setDataType(dataType)
            //    .build();


            // Execution
            cudnnStatus_t status;
            status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << cudnnGetErrorString(status) << std::endl;
            }

            status = cudnnBackendExecute(handle_, poolPlan.get_raw_desc(), variantPackPool.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << cudnnGetErrorString(status) << std::endl;
            }

            status = cudnnBackendExecute(handle_, C3plan.get_raw_desc(), C3variantPack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << std::format("C3: {}", cudnnGetErrorString(status)) << std::endl;
            }

            status = cudnnBackendExecute(handle_, S4poolPlan.get_raw_desc(), S4variantPackPool.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << std::format("S4: {}", cudnnGetErrorString(status)) << std::endl;
            }

            status = cudnnBackendExecute(handle_, FC5plan.get_raw_desc(), FC5variantPack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << std::format("FC5: {}", cudnnGetErrorString(status)) << std::endl;
            }

            status = cudnnBackendExecute(handle_, FC6plan.get_raw_desc(), FC6variantPack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << std::format("FC6: {}", cudnnGetErrorString(status)) << std::endl;
            }

            status = cudnnBackendExecute(handle_, FC7plan.get_raw_desc(), FC7variantPack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << std::format("FC7: {}", cudnnGetErrorString(status)) << std::endl;
            }

            //cudnnSoftmaxForward(handle_, 
            //    CUDNN_SOFTMAX_ACCURATE, 
            //    CUDNN_SOFTMAX_MODE_CHANNEL,
            //    &alpha, 
            //    FC7outputTensor.get_raw_desc(),
            //    FC7devPtrC,
            //    &beta, 
            //    softmaxTensor, 
            //    S.devPtr));


            // Cleanup
            if (FC7workspace_size > 0) {
                checkCudaErr(cudaFree(FC7workspace_ptr));
            }

            if (FC6workspace_size > 0) {
                checkCudaErr(cudaFree(FC6workspace_ptr));
            }

            if (FC5workspace_size > 0) {
                checkCudaErr(cudaFree(FC5workspace_ptr));
            }

            if (S4workspace_size_pool > 0) {
                checkCudaErr(cudaFree(S4workspace_ptr_pool));
            }

            if (C3workspace_size > 0) {
                checkCudaErr(cudaFree(C3workspace_ptr));
            }

            if (workspace_size_pool > 0) {
                checkCudaErr(cudaFree(workspace_ptr_pool));
            }

            if (workspace_size > 0) {
                checkCudaErr(cudaFree(workspace_ptr));
            }

        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }


        // Checking Result

        cudaDeviceSynchronize();
        checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(P.hostPtr, P.devPtr, (size_t)(sizeof(P.hostPtr[0]) * Psize), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(C3Y.hostPtr, C3Y.devPtr, (size_t)(sizeof(C3Y.hostPtr[0]) * C3Y.n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(S4P.hostPtr, S4P.devPtr, (size_t)(sizeof(S4P.hostPtr[0]) * S4P.n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(FC5C.hostPtr, FC5C.devPtr, (size_t)(sizeof(FC5C.hostPtr[0])* FC5C.n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(FC6C.hostPtr, FC6C.devPtr, (size_t)(sizeof(FC6C.hostPtr[0])* FC6C.n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(FC7C.hostPtr, FC7C.devPtr, (size_t)(sizeof(FC7C.hostPtr[0])* FC7C.n_elems), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        display(Y.hostPtr, label, 28);
        display(P.hostPtr, label, 14);
        display(C3Y.hostPtr, label, 10);
        display(S4P.hostPtr, label, S4poolTensorDim[2]);
        display(FC5C.hostPtr, label, 8);
        display(FC6C.hostPtr, label, 8);
        display_flat(FC7C.hostPtr, label, 10);
    }

    void LenetForward_v_6()
    {
        std::cout << "Rigid LeNet Test v0.6" << std::endl;

        using namespace Helpers;

        MNISTDataHolder dh;
        dh.initialize();
        auto [image, label] = dh.getNextTrain();
        auto [rows, cols] = dh.getDimensions();

        //constexpr int64_t N = 1;
        //constexpr int64_t C = 1;
        constexpr int64_t inputH = 28;
        constexpr int64_t inputW = 28;
        constexpr int64_t C1Features = 6;
        constexpr int64_t C3Features = 16;

        // C1 - First convo
        int64_t xTensorDim[] = { 1, 1, 28, 28 }; // input
        int64_t wTensorDim[] = { C1Features, 1, 5, 5 }; // filter
        int64_t yTensorDim[] = { 0, 0, 0, 0 }; // Computed Below
        
        int64_t padA[] = { 2, 2 };
        int64_t dilationA[] = { 1, 1 };
        int64_t convstrideA[] = { 1, 1 };
        int64_t bTensorDim[] = { 1, wTensorDim[0], 1, 1 };  // bias

        yTensorDim[0] = xTensorDim[0];
        yTensorDim[1] = wTensorDim[0];
        for (int dim = 0; dim < 2; dim++) {
            yTensorDim[dim + 2] = Helpers::getFwdConvOutputDim(xTensorDim[dim + 2], padA[dim], wTensorDim[dim + 2], convstrideA[dim], dilationA[dim]);
        }
        
        printf("====DIMENSIONS====\n");
        printf("input dims are %lld, %lld, %lld, %lld\n", xTensorDim[0], xTensorDim[1], xTensorDim[2], xTensorDim[3]);
        printf("filter dims are %lld, %lld, %lld, %lld\n", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]);
        printf("output dims are %lld, %lld, %lld, %lld\n", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]);

        int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

        Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
        Surface<float> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
        Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
        Surface<float> Y(Ysize, false);

        int64_t* x_dim = xTensorDim;
        int64_t* w_dim = wTensorDim;
        int64_t* b_dim = bTensorDim;
        int64_t* y_dim = yTensorDim;

        void* devPtrX = X.devPtr;
        void* devPtrW = W.devPtr;
        void* devPtrY = Y.devPtr;
        void* devPtrB = B.devPtr;

        for (int64_t i = 0; i < X.n_elems; ++i)
        {
            X.hostPtr[i] = static_cast<float>(image[i]) / 256;
            std::cout << std::setw(3) << std::setprecision(3) << X.hostPtr[i] << " ";
            if (i > 1 && ((i + 1) % cols == 0))
            {
                std::cout << std::endl;
            }
        }

        for (int64_t i = 0; i < Y.n_elems; ++i)
        {
            Y.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(devPtrX, X.hostPtr, size_t(sizeof(X.hostPtr[0]) * X.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrY, Y.hostPtr, size_t(sizeof(Y.hostPtr[0]) * Y.n_elems), cudaMemcpyHostToDevice));

        // S2 - Pooling
        int64_t poolTensorDim[] = { 0, 0, 0, 0 };
        poolTensorDim[0] = yTensorDim[0];
        poolTensorDim[1] = yTensorDim[1];
        poolTensorDim[2] = yTensorDim[2] / 2;
        poolTensorDim[3] = yTensorDim[3] / 2;

        int64_t windowDimPool[CUDNN_DIM_MAX] = { 2,2 };
        int64_t prePaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        int64_t postPaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        int64_t stridePool[CUDNN_DIM_MAX] = { 2,2 };

        printf("After pool dims are %lld, %lld, %lld, %lld\n", poolTensorDim[0], poolTensorDim[1], poolTensorDim[2], poolTensorDim[3]);

        
        int64_t Psize = poolTensorDim[0] * poolTensorDim[1] * poolTensorDim[2] * poolTensorDim[3];
        Surface<float> P(Psize, false);

        int64_t* p_dim = poolTensorDim;
        void* devPtrP = P.devPtr;



        for (int64_t i = 0; i < P.n_elems; ++i)
        {
            P.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(devPtrP, P.hostPtr, size_t(sizeof(P.hostPtr[0]) * P.n_elems), cudaMemcpyHostToDevice));

        // C3 - Second convo
        int64_t C3InputDim[] = { poolTensorDim[0], poolTensorDim[1], poolTensorDim[2], poolTensorDim[3]};
        int64_t C3FilterDim[] = { C3Features, C3InputDim[1], 5, 5}; // filter
        int64_t C3OutputDim[] = { 0, 0, 0, 0 }; // Computed Below

        int64_t C3Pad[] = { 0, 0 };
        int64_t C3Dilation[] = { 1, 1 };
        int64_t C3ConvStride[] = { 1, 1 };
        int64_t C3BiasDim[] = { 1, C3FilterDim[0], 1, 1 };  // bias

        C3OutputDim[0] = C3InputDim[0];
        C3OutputDim[1] = C3FilterDim[0];
        for (int dim = 0; dim < 2; dim++) {
            C3OutputDim[dim + 2] = Helpers::getFwdConvOutputDim(C3InputDim[dim + 2], C3Pad[dim], C3FilterDim[dim + 2], C3ConvStride[dim], C3Dilation[dim]);
        }

        printf("====DIMENSIONS====\n");
        printf("C3InputDim dims are %lld, %lld, %lld, %lld\n", C3InputDim[0], C3InputDim[1], C3InputDim[2], C3InputDim[3]);
        printf("C3filter dims are %lld, %lld, %lld, %lld\n", C3FilterDim[0], C3FilterDim[1], C3FilterDim[2], C3FilterDim[3]);
        printf("C3OutputDim dims are %lld, %lld, %lld, %lld\n", C3OutputDim[0], C3OutputDim[1], C3OutputDim[2], C3OutputDim[3]);

        int64_t C3Ysize = C3OutputDim[0] * C3OutputDim[1] * C3OutputDim[2] * C3OutputDim[3];

        Surface<float> C3X(C3InputDim[0] * C3InputDim[1] * C3InputDim[2] * C3InputDim[3], false);
        Surface<float> C3W(C3FilterDim[0] * C3FilterDim[1] * C3FilterDim[2] * C3FilterDim[3], false);
        Surface<float> C3B(C3BiasDim[0] * C3BiasDim[1] * C3BiasDim[2] * C3BiasDim[3], false);
        Surface<float> C3Y(C3Ysize, false);

        int64_t* C3x_dim = C3InputDim;
        int64_t* C3w_dim = C3FilterDim;
        int64_t* C3b_dim = C3BiasDim;
        int64_t* C3y_dim = C3OutputDim;

        void* C3devPtrX = C3X.devPtr;
        void* C3devPtrW = C3W.devPtr;
        void* C3devPtrY = C3Y.devPtr;
        void* C3devPtrB = C3B.devPtr;

        for (int64_t i = 0; i < C3Y.n_elems; ++i)
        {
            C3Y.hostPtr[i] = 1;
        }

        checkCudaErr(cudaMemcpy(C3devPtrY, C3Y.hostPtr, size_t(sizeof(C3Y.hostPtr[0]) * C3Y.n_elems), cudaMemcpyHostToDevice));


        // S4 - Pooling
        int64_t S4poolTensorDim[] = { 0, 0, 0, 0 };
        S4poolTensorDim[0] = C3OutputDim[0];
        S4poolTensorDim[1] = C3OutputDim[1];
        S4poolTensorDim[2] = C3OutputDim[2] / 2;
        S4poolTensorDim[3] = C3OutputDim[3] / 2;

        //int64_t windowDimPool[CUDNN_DIM_MAX] = { 2,2 };
        //int64_t prePaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        //int64_t postPaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        //int64_t stridePool[CUDNN_DIM_MAX] = { 2,2 };

        printf("After pool dims are %lld, %lld, %lld, %lld\n", S4poolTensorDim[0], S4poolTensorDim[1], S4poolTensorDim[2], S4poolTensorDim[3]);


        int64_t S4Psize = S4poolTensorDim[0] * S4poolTensorDim[1] * S4poolTensorDim[2] * S4poolTensorDim[3];
        Surface<float> S4P(S4Psize, false);

        int64_t* S4p_dim = S4poolTensorDim;
        void* S4devPtrP = S4P.devPtr;



        for (int64_t i = 0; i < S4P.n_elems; ++i)
        {
            S4P.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(S4devPtrP, S4P.hostPtr, size_t(sizeof(S4P.hostPtr[0]) * S4P.n_elems), cudaMemcpyHostToDevice));


        checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle_;

        try
        {
            checkCudnnErr(cudnnCreate(&handle_));

            constexpr int64_t alignment = 16; //16
            cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
            cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
            constexpr int convDim = 2;
            float alpha = 1.0f;
            float beta = 0.0f;
            //cudnn_frontend::ExecutionPlan plan;

            // C1 - convolution
            int64_t stride[4];
            generateStrides(x_dim, stride, 4, tensorFormat);
            auto xTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, x_dim)
                .setStride(4, stride)
                .setId('x')
                .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();
            generateStrides(w_dim, stride, 4, tensorFormat);
            auto wTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, w_dim)
                .setStride(4, stride)
                .setId('w')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(b_dim, stride, 4, tensorFormat);
            auto bTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, b_dim)
                .setStride(4, stride)
                .setId('b')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(y_dim, stride, 4, tensorFormat);
            auto afterConvTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('A')  // after conv
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('B')  // after bias
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto yTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('y')  // after relu
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            std::cout << xTensor.describe() << std::endl;
            std::cout << wTensor.describe() << std::endl;
            std::cout << afterConvTensor.describe() << std::endl;
            std::cout << yTensor.describe() << std::endl;

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
            std::cout << convDesc.describe() << std::endl;

            // Define the bias descriptor
            auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_RELU_FWD)//CUDNN_POINTWISE_SIGMOID_FWD
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << actDesc.describe() << std::endl;

            // Create a convolution Node
            auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(xTensor)
                .setwDesc(wTensor)
                .setyDesc(afterConvTensor) //afterConvTensor // yTensor
                .setcDesc(convDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << conv_op.describe() << std::endl;

            // Create a Bias Node.
            auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(conv_op.getOutputTensor()) //conv_op.getOutputTensor()
                .setbDesc(bTensor)
                .setyDesc(afterBiasTensor) //afterBiasTensor
                .setpwDesc(biasDesc)
                .build();
            std::cout << bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(bias_op.getOutputTensor()) // bias_op.getOutputTensor()
                .setyDesc(yTensor)
                .setpwDesc(actDesc)
                .build();
            std::cout << act_op.describe() << std::endl;

            // Create an Operation Graph.
            //std::vector<cudnn_frontend::Operation const*> ops = { &conv_op,  &bias_op, &act_op, &pool_op};
            std::vector<cudnn_frontend::Operation const*> ops;
            ops.emplace_back(&conv_op);
            ops.emplace_back(&bias_op);
            ops.emplace_back(&act_op);

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            std::cout << opGraph.describe() << std::endl;

            auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

            std::cout << "Plan tag: " << plan.getTag() << std::endl;

            int64_t workspace_size = 0;
            void* workspace_ptr = nullptr;

            workspace_size = plan.getWorkspaceSize();
            std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }
            //void* data_ptrs[] = { devPtrX, devPtrW, devPtrY, devPtrB, devPtrP };
            //int64_t uids[] = { 'x', 'w', 'y', 'b', 'p'};
            std::vector<void*> data_ptrs;
            data_ptrs.emplace_back(devPtrX);
            data_ptrs.emplace_back(devPtrW);
            data_ptrs.emplace_back(devPtrB);
            data_ptrs.emplace_back(devPtrY);

            std::vector<int64_t> uids;
            uids.emplace_back('x');
            uids.emplace_back('w');
            uids.emplace_back('b');
            uids.emplace_back('y');

            assert(data_ptrs.size() == uids.size());
            //int64_t num_ptrs = sizeof(uids) / sizeof(uids[0]);
            int64_t num_ptrs = data_ptrs.size();
            std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
            auto variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                //.setDataPointers(num_ptrs, data_ptrs)
                .setDataPointers(num_ptrs, data_ptrs.data())
                .setUids(num_ptrs, uids.data())
                .build();
            std::cout << "variantPack " << variantPack.describe() << std::endl;


            // S2 - Pooling

            auto const nanOpt = CUDNN_NOT_PROPAGATE_NAN;
            constexpr int64_t nbSpatialDims = 2;
            cudnn_frontend::cudnnResampleMode_t const mode = cudnn_frontend::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
            cudnn_frontend::cudnnPaddingMode_t const padding_mode = cudnn_frontend::cudnnPaddingMode_t::CUDNN_ZERO_PAD;

            generateStrides(p_dim, stride, 4, tensorFormat);
            auto afterPoolTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, p_dim)
                .setStride(4, stride)
                .setId('p')  // after pool
                .setAlignment(16)
                .setDataType(dataType)
                .build();
            std::cout << afterPoolTensor.describe() << std::endl;

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
            std::cout << "Initialized Pool Desc" << std::endl;
            std::cout << poolDesc.describe() << std::endl;

            // Create a Resample Node
            auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                .setxDesc(act_op.getOutputTensor())
                .setyDesc(afterPoolTensor)
                .setResampleDesc(poolDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << pool_op.describe() << std::endl;

            std::vector< cudnn_frontend::Operation const*> poolOps = { &pool_op };

            auto poolOpGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(poolOps.size(), poolOps.data())
                .build();

            std::cout << poolOpGraph.describe() << std::endl;

            auto poolPlan = get_execplan_from_heuristics_else_fall_back(std::move(poolOpGraph), handle_);

            std::cout << "poolPlan tag: " << poolPlan.getTag() << std::endl;

            int64_t workspace_size_pool = 0;
            void* workspace_ptr_pool = nullptr;

            workspace_size_pool = poolPlan.getWorkspaceSize();
            std::cout << poolPlan.describe() << " requires workspace " << workspace_size_pool << std::endl;

            if (workspace_size_pool > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr_pool, (size_t)workspace_size_pool));
            }

            std::vector<void*> data_ptrs_pool;
            data_ptrs_pool.emplace_back(devPtrY);
            data_ptrs_pool.emplace_back(devPtrP);

            std::vector<int64_t> uids_pool;
            uids_pool.emplace_back('y');
            uids_pool.emplace_back('p');

            assert(data_ptrs_pool.size() == uids_pool.size());
            int64_t num_ptrs_pool = data_ptrs_pool.size();
            std::cout << std::format("Num ptrs {}", num_ptrs_pool) << std::endl;
            auto variantPackPool = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr_pool)
                .setDataPointers(num_ptrs_pool, data_ptrs_pool.data())
                .setUids(num_ptrs_pool, uids_pool.data())
                .build();
            std::cout << "variantPack " << variantPackPool.describe() << std::endl;


            // C3 - convolution
            int64_t id = 0;
            generateStrides(C3x_dim, stride, 4, tensorFormat);
            auto C3xTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3x_dim)
                .setStride(4, stride)
                .setId(id++) // 0
                .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();

            generateStrides(C3w_dim, stride, 4, tensorFormat);
            auto C3wTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3w_dim)
                .setStride(4, stride)
                .setId(id++) // 1
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(C3b_dim, stride, 4, tensorFormat);
            auto C3bTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3b_dim)
                .setStride(4, stride)
                .setId(id++) // 2
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(C3y_dim, stride, 4, tensorFormat);
            auto C3yTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3y_dim)
                .setStride(4, stride)
                .setId(id++)  // after relu // 3
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            auto C3afterConvTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3y_dim)
                .setStride(4, stride)
                .setId(id++)  // after conv // 4
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto C3afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, C3y_dim)
                .setStride(4, stride)
                .setId(id++)  // after bias // 5
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            std::cout << C3xTensor.describe() << std::endl;
            std::cout << C3wTensor.describe() << std::endl;
            std::cout << C3bTensor.describe() << std::endl;
            std::cout << C3yTensor.describe() << std::endl;
            std::cout << C3afterConvTensor.describe() << std::endl;
            std::cout << C3afterBiasTensor.describe() << std::endl;

            // Define the convolution problem
            auto C3convDesc = cudnn_frontend::ConvDescBuilder()
                .setComputeType(CUDNN_DATA_FLOAT)
                .setMathMode(CUDNN_CROSS_CORRELATION)
                .setSpatialDimCount(convDim)
                .setSpatialStride(convDim, C3ConvStride)
                .setPrePadding(convDim, C3Pad)
                .setPostPadding(convDim, C3Pad)
                .setDilation(convDim, C3Dilation)
                .build();
            std::cout << C3convDesc.describe() << std::endl;

            // Define the bias descriptor
            auto C3biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << C3biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto C3actDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_RELU_FWD)//CUDNN_POINTWISE_SIGMOID_FWD
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << C3actDesc.describe() << std::endl;

            // Create a convolution Node
            auto C3conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(pool_op.getOutputTensor()) //pool_op.getOutputTensor()
                .setwDesc(C3wTensor)
                .setyDesc(C3afterConvTensor) //C3afterConvTensor // C3yTensor
                .setcDesc(C3convDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << C3conv_op.describe() << std::endl;

            // Create a Bias Node.
            auto C3bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(C3conv_op.getOutputTensor())
                .setbDesc(C3bTensor)
                .setyDesc(C3afterBiasTensor) //afterBiasTensor
                .setpwDesc(C3biasDesc)
                .build();
            std::cout << C3bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto C3act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(C3bias_op.getOutputTensor()) // bias_op.getOutputTensor()
                .setyDesc(C3yTensor)
                .setpwDesc(C3actDesc)
                .build();
            std::cout << C3act_op.describe() << std::endl;

            // Create an Operation Graph.
            std::vector<cudnn_frontend::Operation const*> C3ops;
            C3ops.emplace_back(&C3conv_op);
            C3ops.emplace_back(&C3bias_op);
            C3ops.emplace_back(&C3act_op);

            auto C3opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(C3ops.size(), C3ops.data())
                .build();

            std::cout << C3opGraph.describe() << std::endl;

            auto C3plan = get_execplan_from_heuristics_else_fall_back(std::move(C3opGraph), handle_);

            std::cout << "C3Plan tag: " << C3plan.getTag() << std::endl;

            auto C3workspace_size = C3plan.getWorkspaceSize();
            std::cout << C3plan.describe() << " requires workspace " << C3workspace_size << std::endl;

            void* C3workspace_ptr = nullptr;
            if (C3workspace_size > 0) {
                checkCudaErr(cudaMalloc(&C3workspace_ptr, (size_t)C3workspace_size));
            }
            std::vector<void*> C3data_ptrs;
            //C3data_ptrs.emplace_back(C3devPtrX);
            C3data_ptrs.emplace_back(devPtrP);
            C3data_ptrs.emplace_back(C3devPtrW);
            C3data_ptrs.emplace_back(C3devPtrB);
            C3data_ptrs.emplace_back(C3devPtrY);

            std::vector<int64_t> C3uids;
            //C3uids.emplace_back(0);
            C3uids.emplace_back('p');
            C3uids.emplace_back(1);
            C3uids.emplace_back(2);
            C3uids.emplace_back(3);

            assert(C3data_ptrs.size() == C3uids.size());
            int64_t C3num_ptrs = C3data_ptrs.size();
            std::cout << std::format("Num ptrs {}", C3num_ptrs) << std::endl;
            auto C3variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(C3workspace_ptr)
                .setDataPointers(C3num_ptrs, C3data_ptrs.data())
                .setUids(C3num_ptrs, C3uids.data())
                .build();
            std::cout << "variantPack " << C3variantPack.describe() << std::endl;


            // S4 - Pooling
            generateStrides(S4p_dim, stride, 4, tensorFormat);
            auto S4afterPoolTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, S4p_dim)
                .setStride(4, stride)
                .setId(id++)  // after pool
                .setAlignment(16)
                .setDataType(dataType)
                .build();
            std::cout << afterPoolTensor.describe() << std::endl;

            // Define the resample descriptor
            auto S4poolDesc = cudnn_frontend::ResampleDescBuilder()
                .setComputeType(CUDNN_DATA_FLOAT)
                .setNanPropagation(nanOpt)
                .setResampleMode(mode)
                .setPaddingMode(padding_mode)
                .setSpatialDim(nbSpatialDims, windowDimPool)
                .setSpatialStride(nbSpatialDims, stridePool)
                .setPrePadding(nbSpatialDims, prePaddingPool)
                .setPostPadding(nbSpatialDims, postPaddingPool)
                .build();
            std::cout << "Initialized Pool Desc" << std::endl;
            std::cout << S4poolDesc.describe() << std::endl;

            // Create a Resample Node
            auto S4pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                .setxDesc(C3act_op.getOutputTensor())
                .setyDesc(S4afterPoolTensor)
                .setResampleDesc(S4poolDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << S4pool_op.describe() << std::endl;

            std::vector< cudnn_frontend::Operation const*> S4poolOps = { &S4pool_op };

            auto S4poolOpGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(S4poolOps.size(), S4poolOps.data())
                .build();

            std::cout << S4poolOpGraph.describe() << std::endl;

            auto S4poolPlan = get_execplan_from_heuristics_else_fall_back(std::move(S4poolOpGraph), handle_);

            std::cout << "poolPlan tag: " << S4poolPlan.getTag() << std::endl;

            int64_t S4workspace_size_pool = 0;
            void* S4workspace_ptr_pool = nullptr;

            S4workspace_size_pool = poolPlan.getWorkspaceSize();
            std::cout << S4poolPlan.describe() << " requires workspace " << S4workspace_size_pool << std::endl;

            if (S4workspace_size_pool > 0) {
                checkCudaErr(cudaMalloc(&S4workspace_ptr_pool, (size_t)S4workspace_size_pool));
            }

            std::vector<void*> S4data_ptrs_pool;
            S4data_ptrs_pool.emplace_back(C3devPtrY);
            S4data_ptrs_pool.emplace_back(S4devPtrP);

            std::vector<int64_t> S4uids_pool;
            S4uids_pool.emplace_back(C3yTensor.getId());
            S4uids_pool.emplace_back(S4afterPoolTensor.getId());

            assert(S4data_ptrs_pool.size() == S4uids_pool.size());
            int64_t S4num_ptrs_pool = S4data_ptrs_pool.size();
            std::cout << std::format("Num ptrs {}", S4num_ptrs_pool) << std::endl;
            auto S4variantPackPool = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(S4workspace_ptr_pool)
                .setDataPointers(S4num_ptrs_pool, S4data_ptrs_pool.data())
                .setUids(S4num_ptrs_pool, S4uids_pool.data())
                .build();
            std::cout << "variantPack " << S4variantPackPool.describe() << std::endl;


            // Execution
            cudnnStatus_t status;
            status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << cudnnGetErrorString(status) << std::endl;
            }
            
            status = cudnnBackendExecute(handle_, poolPlan.get_raw_desc(), variantPackPool.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << cudnnGetErrorString(status) << std::endl;
            }

            status = cudnnBackendExecute(handle_, C3plan.get_raw_desc(), C3variantPack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << std::format("C3: {}", cudnnGetErrorString(status)) << std::endl;
            }

            status = cudnnBackendExecute(handle_, S4poolPlan.get_raw_desc(), S4variantPackPool.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS)
            {
                std::cout << std::format("S4: {}", cudnnGetErrorString(status)) << std::endl;
            }



            // Cleanup
            if (S4workspace_size_pool > 0) {
                checkCudaErr(cudaFree(S4workspace_ptr_pool));
            }

            if (C3workspace_size > 0) {
                checkCudaErr(cudaFree(C3workspace_ptr));
            }

            if (workspace_size_pool > 0) {
                checkCudaErr(cudaFree(workspace_ptr_pool));
            }

            if (workspace_size > 0) {
                checkCudaErr(cudaFree(workspace_ptr));
            }

        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }


        // Checking Result

        cudaDeviceSynchronize();
        checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(P.hostPtr, P.devPtr, (size_t)(sizeof(P.hostPtr[0]) * Psize), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(C3Y.hostPtr, C3Y.devPtr, (size_t)(sizeof(C3Y.hostPtr[0])* C3Y.n_elems), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(S4P.hostPtr, S4P.devPtr, (size_t)(sizeof(S4P.hostPtr[0])* S4P.n_elems), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        display(Y.hostPtr, label, 28);
        display(P.hostPtr, label, 14);
        display(C3Y.hostPtr, label, 10);
        display(S4P.hostPtr, label, S4poolTensorDim[2]);
    }

    void LenetForward_v_5()
    {
        std::cout << "Rigid LeNet Test v0.5" << std::endl;

        using namespace Helpers;

        MNISTDataHolder dh;
        dh.initialize();
        auto [image, label] = dh.getNextTrain();
        auto [rows, cols] = dh.getDimensions();

        //constexpr int64_t N = 1;
        //constexpr int64_t C = 1;
        constexpr int64_t inputH = 28;
        constexpr int64_t inputW = 28;
        constexpr int64_t C1Features = 6;
        constexpr int64_t C3Features = 16;

        int64_t xTensorDim[] = { 1, 1, 28, 28 }; // input
        int64_t wTensorDim[] = { C1Features, 1, 5, 5 }; // filter
        int64_t yTensorDim[] = { 0, 0, 0, 0 }; // Computed Below
        int64_t poolTensorDim[] = { 0, 0, 0, 0 }; // Computed Below
        int64_t padA[] = { 2, 2 };
        int64_t dilationA[] = { 1, 1 };
        int64_t convstrideA[] = { 1, 1 };
        int64_t bTensorDim[] = { 1, wTensorDim[0], 1, 1 };  // bias

        yTensorDim[0] = xTensorDim[0];
        yTensorDim[1] = wTensorDim[0];
        for (int dim = 0; dim < 2; dim++) {
            yTensorDim[dim + 2] = Helpers::getFwdConvOutputDim(xTensorDim[dim + 2], padA[dim], wTensorDim[dim + 2], convstrideA[dim], dilationA[dim]);
        }
        poolTensorDim[0] = yTensorDim[0];
        poolTensorDim[1] = yTensorDim[1];
        poolTensorDim[2] = yTensorDim[2] / 2;
        poolTensorDim[3] = yTensorDim[3] / 2;

        int64_t windowDimPool[CUDNN_DIM_MAX] = { 2,2 };
        int64_t prePaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        int64_t postPaddingPool[CUDNN_DIM_MAX] = { 0,0 };
        int64_t stridePool[CUDNN_DIM_MAX] = { 2,2 };


        printf("====DIMENSIONS====\n");
        printf("input dims are %lld, %lld, %lld, %lld\n", xTensorDim[0], xTensorDim[1], xTensorDim[2], xTensorDim[3]);
        printf("filter dims are %lld, %lld, %lld, %lld\n", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]);
        printf("output dims are %lld, %lld, %lld, %lld\n", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]);
        printf("After pool dims are %lld, %lld, %lld, %lld\n", poolTensorDim[0], poolTensorDim[1], poolTensorDim[2], poolTensorDim[3]);

        int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];
        int64_t Psize = poolTensorDim[0] * poolTensorDim[1] * poolTensorDim[2] * poolTensorDim[3];

        Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
        Surface<float> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
        Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
        Surface<float> Y(Ysize, false);
        Surface<float> P(Psize, false);

        int64_t* x_dim = xTensorDim;
        int64_t* w_dim = wTensorDim;
        int64_t* b_dim = bTensorDim;
        int64_t* y_dim = yTensorDim;
        int64_t* p_dim = poolTensorDim;
        void* devPtrX = X.devPtr;
        void* devPtrW = W.devPtr;
        void* devPtrY = Y.devPtr;
        void* devPtrB = B.devPtr;
        void* devPtrP = P.devPtr;

        for (int64_t i = 0; i < X.n_elems; ++i)
        {
            X.hostPtr[i] = static_cast<float>(image[i]) / 256;
            std::cout << std::setw(3) << std::setprecision(3) << X.hostPtr[i] << " ";
            if (i > 1 && ((i + 1) % cols == 0))
            {
                std::cout << std::endl;
            }
        }

        for (int64_t i = 0; i < Y.n_elems; ++i)
        {
            Y.hostPtr[i] = 0;
        }

        for (int64_t i = 0; i < P.n_elems; ++i)
        {
            P.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(devPtrX, X.hostPtr, size_t(sizeof(X.hostPtr[0]) * X.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrY, Y.hostPtr, size_t(sizeof(Y.hostPtr[0]) * Y.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrP, P.hostPtr, size_t(sizeof(P.hostPtr[0]) * P.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle_;

        try
        {
            checkCudnnErr(cudnnCreate(&handle_));

            constexpr int64_t alignment = 16; //16
            cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
            cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
            constexpr int convDim = 2;
            auto const nanOpt = CUDNN_NOT_PROPAGATE_NAN;
            constexpr int64_t nbSpatialDims = 2;
            cudnn_frontend::cudnnResampleMode_t const mode = cudnn_frontend::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
            cudnn_frontend::cudnnPaddingMode_t const padding_mode = cudnn_frontend::cudnnPaddingMode_t::CUDNN_ZERO_PAD;

            // Creates the necessary tensor descriptors
            int64_t stride[4];
            generateStrides(x_dim, stride, 4, tensorFormat);
            auto xTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, x_dim)
                .setStride(4, stride)
                .setId('x')
                .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();
            generateStrides(w_dim, stride, 4, tensorFormat);
            auto wTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, w_dim)
                .setStride(4, stride)
                .setId('w')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(b_dim, stride, 4, tensorFormat);
            auto bTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, b_dim)
                .setStride(4, stride)
                .setId('b')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(y_dim, stride, 4, tensorFormat);
            auto afterConvTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('A')  // after conv
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('B')  // after bias
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto yTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('y')  // after relu
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(p_dim, stride, 4, tensorFormat);
            auto afterPoolTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, p_dim)
                .setStride(4, stride)
                .setId('p')  // after pool
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            std::cout << xTensor.describe() << std::endl;
            std::cout << wTensor.describe() << std::endl;
            std::cout << afterConvTensor.describe() << std::endl;
            std::cout << yTensor.describe() << std::endl;
            std::cout << afterPoolTensor.describe() << std::endl;


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
            std::cout << convDesc.describe() << std::endl;

            // Define the bias descriptor
            auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_RELU_FWD)//CUDNN_POINTWISE_SIGMOID_FWD
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << actDesc.describe() << std::endl;

            //// Define the resample descriptor
            //auto poolDesc = cudnn_frontend::ResampleDescBuilder()
            //    .setComputeType(CUDNN_DATA_FLOAT)
            //    .setNanPropagation(nanOpt)
            //    .setResampleMode(mode)
            //    .setPaddingMode(padding_mode)
            //    .setSpatialDim(nbSpatialDims, windowDimPool)
            //    .setSpatialStride(nbSpatialDims, stridePool)
            //    .setPrePadding(nbSpatialDims, prePaddingPool)
            //    .setPostPadding(nbSpatialDims, postPaddingPool)
            //    .build();
            //std::cout << "Initialized Pool Desc" << std::endl;
            //std::cout << poolDesc.describe() << std::endl;

            float alpha = 1.0f;
            float beta = 0.0f;

            // Create a convolution Node
            auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(xTensor)
                .setwDesc(wTensor)
                .setyDesc(afterConvTensor) //afterConvTensor // yTensor
                .setcDesc(convDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << conv_op.describe() << std::endl;

            // Create a Bias Node.
            auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(conv_op.getOutputTensor()) //conv_op.getOutputTensor()
                .setbDesc(bTensor)
                .setyDesc(afterBiasTensor) //afterBiasTensor
                .setpwDesc(biasDesc)
                .build();
            std::cout << bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(bias_op.getOutputTensor()) // bias_op.getOutputTensor()
                .setyDesc(yTensor)
                .setpwDesc(actDesc)
                .build();
            std::cout << act_op.describe() << std::endl;

            //// Create a Resample Node
            //auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
            //    .setxDesc(act_op.getOutputTensor()) // act_op.getOutputTensor()
            //    .setyDesc(afterPoolTensor)
            //    .setResampleDesc(poolDesc)
            //    .setAlpha(alpha)
            //    .setBeta(beta)
            //    .build();
            //std::cout << pool_op.describe() << std::endl;

            // Create an Operation Graph.
            //std::vector<cudnn_frontend::Operation const*> ops = { &conv_op,  &bias_op, &act_op, &pool_op};
            std::vector<cudnn_frontend::Operation const*> ops;
            ops.emplace_back(&conv_op);
            ops.emplace_back(&bias_op);
            ops.emplace_back(&act_op);
            //ops.emplace_back(&pool_op);

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            std::cout << opGraph.describe() << std::endl;

            auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

            std::cout << "Plan tag: " << plan.getTag() << std::endl;

            auto workspace_size = plan.getWorkspaceSize();
            std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

            void* workspace_ptr = nullptr;
            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }
            //void* data_ptrs[] = { devPtrX, devPtrW, devPtrY, devPtrB, devPtrP };
            //int64_t uids[] = { 'x', 'w', 'y', 'b', 'p'};
            std::vector<void*> data_ptrs;
            data_ptrs.emplace_back(devPtrX);
            data_ptrs.emplace_back(devPtrW);
            data_ptrs.emplace_back(devPtrB);
            data_ptrs.emplace_back(devPtrY);
            //data_ptrs.emplace_back(devPtrP);

            std::vector<int64_t> uids;
            uids.emplace_back('x');
            uids.emplace_back('w');
            uids.emplace_back('b');
            uids.emplace_back('y');
            //uids.emplace_back('p');

            assert(data_ptrs.size() == uids.size());
            //int64_t num_ptrs = sizeof(uids) / sizeof(uids[0]);
            int64_t num_ptrs = data_ptrs.size();
            std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
            auto variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                //.setDataPointers(num_ptrs, data_ptrs)
                .setDataPointers(num_ptrs, data_ptrs.data())
                .setUids(num_ptrs, uids.data())
                .build();
            std::cout << "variantPack " << variantPack.describe() << std::endl;
            cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());


            // Pooling

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
            std::cout << "Initialized Pool Desc" << std::endl;
            std::cout << poolDesc.describe() << std::endl;

            // Create a Resample Node
            auto pool_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                .setxDesc(act_op.getOutputTensor()) // act_op.getOutputTensor()
                .setyDesc(afterPoolTensor)
                .setResampleDesc(poolDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << pool_op.describe() << std::endl;

            std::vector< cudnn_frontend::Operation const*> poolOps = { &pool_op };

            auto poolOpGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(poolOps.size(), poolOps.data())
                .build();

            std::cout << poolOpGraph.describe() << std::endl;

            auto poolPlan = get_execplan_from_heuristics_else_fall_back(std::move(poolOpGraph), handle_);

            std::cout << "poolPlan tag: " << poolPlan.getTag() << std::endl;

            auto workspace_size_pool = poolPlan.getWorkspaceSize();
            std::cout << poolPlan.describe() << " requires workspace " << workspace_size_pool << std::endl;

            void* workspace_ptr_pool = nullptr;
            if (workspace_size_pool > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr_pool, (size_t)workspace_size_pool));
            }

            std::vector<void*> data_ptrs_pool;
            data_ptrs_pool.emplace_back(devPtrY);
            data_ptrs_pool.emplace_back(devPtrP);

            std::vector<int64_t> uids_pool;
            uids_pool.emplace_back('y');
            uids_pool.emplace_back('p');

            assert(data_ptrs_pool.size() == uids_pool.size());
            int64_t num_ptrs_pool = data_ptrs_pool.size();
            std::cout << std::format("Num ptrs {}", num_ptrs_pool) << std::endl;
            auto variantPackPool = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr_pool)
                .setDataPointers(num_ptrs_pool, data_ptrs_pool.data())
                .setUids(num_ptrs_pool, uids_pool.data())
                .build();
            std::cout << "variantPack " << variantPackPool.describe() << std::endl;
            status = cudnnBackendExecute(handle_, poolPlan.get_raw_desc(), variantPackPool.get_raw_desc());

            // Cleanup
            if (workspace_size_pool > 0) {
                checkCudaErr(cudaFree(workspace_ptr_pool));
            }

            if (workspace_size > 0) {
                checkCudaErr(cudaFree(workspace_ptr));
            }

        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }

        cudaDeviceSynchronize();
        checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
        checkCudaErr(cudaMemcpy(P.hostPtr, P.devPtr, (size_t)(sizeof(P.hostPtr[0])* Psize), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        display(Y.hostPtr, label, 28);
        display(P.hostPtr, label, 14);
    }

    export void LenetForward_v_4()
    {
        std::cout << "Rigid LeNet Test v0.4" << std::endl;

        using namespace Helpers;

        MNISTDataHolder dh;
        dh.initialize();
        auto [image, label] = dh.getNextTrain();
        auto [rows, cols] = dh.getDimensions();

        //constexpr int64_t N = 1;
        //constexpr int64_t C = 1;
        //constexpr int64_t H = 28;
        //constexpr int64_t W = 28;

        int64_t xTensorDim[] = { 1, 6, 14, 14 }; // input
        int64_t wTensorDim[] = { 16, xTensorDim[1], 5, 5}; // filter
        int64_t yTensorDim[] = { 0, 0, 0, 0 }; // Computed Below
        int64_t padA[] = { 0, 0 };
        int64_t dilationA[] = { 1, 1 };
        int64_t convstrideA[] = { 1, 1 };
        int64_t bTensorDim[] = { 1, wTensorDim[0], 1, 1};  // bias

        yTensorDim[0] = xTensorDim[0];
        yTensorDim[1] = wTensorDim[0];
        for (int dim = 0; dim < 2; dim++) {
            yTensorDim[dim + 2] = Helpers::getFwdConvOutputDim(xTensorDim[dim + 2], padA[dim], wTensorDim[dim + 2], convstrideA[dim], dilationA[dim]);
        }

        printf("====DIMENSIONS====\n");
        printf("input dims are %lld, %lld, %lld, %lld\n", xTensorDim[0], xTensorDim[1], xTensorDim[2], xTensorDim[3]);
        printf("filter dims are %lld, %lld, %lld, %lld\n", wTensorDim[0], wTensorDim[1], wTensorDim[2], wTensorDim[3]);
        printf("output dims are %lld, %lld, %lld, %lld\n", yTensorDim[0], yTensorDim[1], yTensorDim[2], yTensorDim[3]);

        int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

        Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
        Surface<float> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
        Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
        Surface<float> Y(Ysize, false);

        int64_t* x_dim = xTensorDim;
        int64_t* w_dim = wTensorDim;
        int64_t* b_dim = bTensorDim;
        int64_t* y_dim = yTensorDim;
        void* devPtrX = X.devPtr;
        void* devPtrW = W.devPtr;
        void* devPtrY = Y.devPtr;
        void* devPtrB = B.devPtr;

        //for (int64_t i = 0; i < X.n_elems; ++i)
        //{
        //    X.hostPtr[i] = static_cast<float>(image[i]) / 256;
        //    std::cout << std::setprecision(3) << X.hostPtr[i];
        //    if (i > 1 && ((i - 1) % 14) == 0)
        //    {
        //        std::cout << std::endl;
        //    }
        //}

        for (int64_t i = 0; i < Y.n_elems; ++i)
        {
            Y.hostPtr[i] = 0;
        }

        checkCudaErr(cudaMemcpy(devPtrX, X.hostPtr, size_t(sizeof(X.hostPtr[0]) * X.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrY, Y.hostPtr, size_t(sizeof(Y.hostPtr[0]) * Y.n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle_;

        try
        {
            checkCudnnErr(cudnnCreate(&handle_));

            constexpr int64_t alignment = 16; //16
            cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
            cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
            constexpr int convDim = 2;

            // Creates the necessary tensor descriptors
            int64_t stride[4];
            generateStrides(x_dim, stride, 4, tensorFormat);
            auto xTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, x_dim)
                .setStride(4, stride)
                .setId('x')
                .setAlignment(alignment)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();
            generateStrides(w_dim, stride, 4, tensorFormat);
            auto wTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, w_dim)
                .setStride(4, stride)
                .setId('w')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(b_dim, stride, 4, tensorFormat);
            auto bTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, b_dim)
                .setStride(4, stride)
                .setId('b')
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            generateStrides(y_dim, stride, 4, tensorFormat);
            auto afterConvTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('A')  // after conv
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('B')  // after bias
                .setAlignment(alignment)
                .setVirtual()
                .setDataType(dataType)
                .build();

            auto yTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('y')  // output
                .setAlignment(alignment)
                .setDataType(dataType)
                .build();

            std::cout << xTensor.describe() << std::endl;
            std::cout << wTensor.describe() << std::endl;
            std::cout << afterConvTensor.describe() << std::endl;
            std::cout << yTensor.describe() << std::endl;


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
            std::cout << convDesc.describe() << std::endl;

            // Define the bias descriptor
            auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << biasDesc.describe() << std::endl;

            // Define the activation descriptor
            auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_RELU_FWD)//CUDNN_POINTWISE_SIGMOID_FWD
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << actDesc.describe() << std::endl;

            float alpha = 1.0f;
            float beta = 0.0f;

            // Create a convolution Node
            auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(xTensor)
                .setwDesc(wTensor)
                .setyDesc(afterConvTensor)
                .setcDesc(convDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << conv_op.describe() << std::endl;

            // Create a Bias Node.
            auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(conv_op.getOutputTensor())
                .setbDesc(bTensor)
                .setyDesc(afterBiasTensor)
                .setpwDesc(biasDesc)
                .build();
            std::cout << bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(bias_op.getOutputTensor())
                .setyDesc(yTensor)
                .setpwDesc(actDesc)
                .build();
            std::cout << act_op.describe() << std::endl;

            // Create an Operation Graph.
            std::vector<cudnn_frontend::Operation const*> ops = { &conv_op, &bias_op, &act_op };

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            std::cout << opGraph.describe() << std::endl;

            auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

            std::cout << "Plan tag: " << plan.getTag() << std::endl;

            auto workspace_size = plan.getWorkspaceSize();
            std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

            void* workspace_ptr = nullptr;
            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }
            void* data_ptrs[] = { devPtrX, devPtrY, devPtrW, devPtrB};
            int64_t uids[] = { 'x', 'y', 'w', 'b'};
            int64_t num_ptrs = sizeof(uids) / sizeof(uids[0]);
            std::cout << std::format("Num ptrs {}", num_ptrs) << std::endl;
            auto variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                .setDataPointers(num_ptrs, data_ptrs)
                .setUids(num_ptrs, uids)
                .build();
            std::cout << "variantPack " << variantPack.describe() << std::endl;
            cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
            if (workspace_size > 0) {
                checkCudaErr(cudaFree(workspace_ptr));
            }

        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }

        cudaDeviceSynchronize();
        checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0]) * Ysize), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        display(Y.hostPtr, label, yTensorDim[2]);
    }

    export void LenetForward_v_3()
    {
        std::cout << "Rigid LeNet Test v0.3" << std::endl;

        using namespace Helpers;

        MNISTDataHolder dh;
        dh.initialize();
        auto [image, label] = dh.getNextTrain();
        auto [rows, cols] = dh.getDimensions();

        //int64_t dimA[] = { 1, 1, 28, 28 };
        //int64_t filterdimA[] = { 1, 1, 5, 5 };
        //int64_t outdimA[] = { 0, 0, 0, 0 }; // Computed Below
        //int64_t padA[] = { 2, 2 };
        //int64_t dilationA[] = { 1, 1 };
        //int64_t convstrideA[] = { 1, 1 };

        //outdimA[0] = dimA[0];
        //outdimA[1] = filterdimA[0];
        //for (int dim = 0; dim < 2; dim++) {
        //    outdimA[dim + 2] = Helpers::getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
        //}

        //cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

        //printf("====DIMENSIONS====\n");
        //printf("input dims are %lld, %lld, %lld, %lld\n", dimA[0], dimA[1], dimA[2], dimA[3]);
        //printf("filter dims are %lld, %lld, %lld, %lld\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
        //printf("output dims are %lld, %lld, %lld, %lld\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


        //int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
        //int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
        //int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];
        //int64_t Bsize = outdimA[0] * outdimA[1] * 1 * 1;

        //run_serialization_conv_bias_scale_relu

        int64_t xTensorDim[] = { 1, 16, 512, 512 };
        int64_t wTensorDim[] = { 64, 16, 3, 3 };
        int64_t yTensorDim[] = { 1, 64, 512, 512 };

        int64_t conv_padA[] = { 1, 1 };
        int64_t conv_dilationA[] = { 1, 1 };
        int64_t conv_strideA[] = { 1, 1 };

        int64_t bTensorDim[] = { 1, 64, 1, 1 };  // bias
        int64_t sTensorDim[] = { 1, 64, 1, 1 };  // scale

        //int64_t xTensorDim[] = { 1, 1, 28, 28 };
        //int64_t wTensorDim[] = { 1, 1, 5, 5 };
        //int64_t yTensorDim[] = { 1, 1, 28, 28 };

        //int64_t conv_padA[] = { 2, 2 };
        //int64_t conv_dilationA[] = { 1, 1 };
        //int64_t conv_strideA[] = { 1, 1 };

        //int64_t bTensorDim[] = { 1, 1, 28, 28 };  // bias
        //int64_t sTensorDim[] = { 1, 1, 28, 28 };  // scale

        printf("====DIMENSIONS====\n");
        printf("input dims are %lld, %lld, %lld, %lld\n",
            xTensorDim[0],
            xTensorDim[1],
            xTensorDim[2],
            xTensorDim[3]);
        printf("filter dims are %lld, %lld, %lld, %lld\n",
            wTensorDim[0],
            wTensorDim[1],
            wTensorDim[2],
            wTensorDim[3]);
        printf("output dims are %lld, %lld, %lld, %lld\n",
            yTensorDim[0],
            yTensorDim[1],
            yTensorDim[2],
            yTensorDim[3]);

        int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

        Surface<float> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
        Surface<float> W(wTensorDim[0] * wTensorDim[1] * wTensorDim[2] * wTensorDim[3], false);
        Surface<float> Y(Ysize, true);

        Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
        Surface<float> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);

        // run_serialization_conv_bias_scale_relu
        int64_t* x_dim = xTensorDim;
        int64_t* w_dim = wTensorDim;
        int64_t* y_dim = yTensorDim;
        int64_t* b_dim = bTensorDim;
        int64_t* s_dim = sTensorDim;
        cudnnDataType_t dataType = CUDNN_DATA_HALF;
        int convDim = 2;
        //int64_t* conv_padA = conv_padA;
        //int64_t* conv_dilationA = conv_dilationA;
        //int64_t* conv_strideA = conv_strideA;
        void* devPtrX = X.devPtr;
        void* devPtrW = W.devPtr;
        void* devPtrY = Y.devPtr;
        void* devPtrB = B.devPtr;
        void* devPtrS = S.devPtr;

        //display(X.hostPtr, X.n_elems);
        //display(W.hostPtr, W.n_elems);
        //display(Y.hostPtr, Y.n_elems);

        //display(B.hostPtr, B.n_elems);
        //display(S.hostPtr, S.n_elems);

        //initImagev2(B.hostPtr, B.n_elems);
        //initImagev2(S.hostPtr, S.n_elems);

        //display(B.hostPtr, B.n_elems);
        //display(S.hostPtr, S.n_elems);


        //// Pool
        //int64_t xTensorDim[] = { 16, 16, 32, 32 };
        //int64_t yTensorDim[] = { 16, 16, 16, 16 };
        //int64_t bTensorDim[] = { 1, 16, 1, 1 };  // bias
        //int64_t sTensorDim[] = { 1, 16, 1, 1 };  // scale

        //cudnnDataType_t compType = CUDNN_DATA_FLOAT;
        //auto const nanOpt = CUDNN_NOT_PROPAGATE_NAN;
        //cudnn_frontend::cudnnResampleMode_t const mode = cudnn_frontend::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
        //cudnn_frontend::cudnnPaddingMode_t const padding_mode = cudnn_frontend::cudnnPaddingMode_t::CUDNN_ZERO_PAD;

        //int64_t nbSpatialDims = 2;
        //double alpha = 1.0;
        //double beta = 0.0;

        //int64_t windowDimA[CUDNN_DIM_MAX] = { 2,2 };
        //int64_t prePaddingA[CUDNN_DIM_MAX] = { 0,0 };
        //int64_t postPaddingA[CUDNN_DIM_MAX] = { 0,0 };
        //int64_t strideA[CUDNN_DIM_MAX] = { 2,2 };

        //printf("====DIMENSIONS====\n");
        //printf("input dims are %lld, %lld, %lld, %lld\n",
        //    xTensorDim[0],
        //    xTensorDim[1],
        //    xTensorDim[2],
        //    xTensorDim[3]);

        //printf("output dims are %lld, %lld, %lld, %lld\n",
        //    yTensorDim[0],
        //    yTensorDim[1],
        //    yTensorDim[2],
        //    yTensorDim[3]);

        //int64_t Ysize = yTensorDim[0] * yTensorDim[1] * yTensorDim[2] * yTensorDim[3];

        //Surface<int8_t> X(xTensorDim[0] * xTensorDim[1] * xTensorDim[2] * xTensorDim[3], false);
        //Surface<int8_t> Y(Ysize, true);

        //Surface<float> B(bTensorDim[0] * bTensorDim[1] * bTensorDim[2] * bTensorDim[3], false);
        //Surface<float> S(sTensorDim[0] * sTensorDim[1] * sTensorDim[2] * sTensorDim[3], false);

        ////run_pool_scale_bias_relu_int8
        //xTensorDim;
        //yTensorDim;
        //bTensorDim;
        //sTensorDim;
        //X.devPtr;
        //Y.devPtr;
        //B.devPtr;
        //S.devPtr;
        //compType;
        //nanOpt;
        //mode;
        //padding_mode;
        //nbSpatialDims;
        //alpha;
        //beta;
        //windowDimA;
        //prePaddingA;
        //postPaddingA;
        //strideA;


        // End pool




        //float* devPtrX = NULL;
        //float* devPtrW = NULL;
        //float* devPtrY = NULL;
        //float* devPtrZ = NULL;
        //float* devPtrB = NULL;
        //float* devPtrAfterAdd = NULL;
        //float* devPtrAfterConv = NULL;
        //float* devPtrAfterBias = NULL;

        //float* hostX = NULL;
        //float* hostW = NULL;
        //float* hostY = NULL;
        //float* hostZ = NULL;
        //float* hostB = NULL;
        //float* hostAfterAdd = NULL;
        //float* hostAfterConv = NULL;
        //float* hostAfterBias = NULL;

        //checkCudaErr(cudaMalloc((void**)&(devPtrX), size_t((Xsize) * sizeof(devPtrX[0]))));
        //checkCudaErr(cudaMalloc((void**)&(devPtrW), size_t((Wsize) * sizeof(devPtrW[0]))));
        //checkCudaErr(cudaMalloc((void**)&(devPtrY), size_t((Ysize) * sizeof(devPtrY[0]))));
        //checkCudaErr(cudaMalloc((void**)&(devPtrZ), size_t((Ysize) * sizeof(devPtrZ[0]))));
        //checkCudaErr(cudaMalloc((void**)&(devPtrB), size_t((Bsize) * sizeof(devPtrB[0]))));
        //checkCudaErr(cudaMalloc((void**)&(devPtrAfterConv), size_t((Ysize) * sizeof(devPtrAfterConv[0]))));
        //checkCudaErr(cudaMalloc((void**)&(devPtrAfterAdd), size_t((Ysize) * sizeof(devPtrAfterAdd[0]))));
        //checkCudaErr(cudaMalloc((void**)&(devPtrAfterBias), size_t((Ysize) * sizeof(devPtrAfterBias[0]))));

        //hostX = (float*)calloc(size_t(Xsize), sizeof(hostX[0]));
        //hostW = (float*)calloc(size_t(Wsize), sizeof(hostW[0]));
        //hostY = (float*)calloc(size_t(Ysize), sizeof(hostY[0]));
        //hostZ = (float*)calloc(size_t(Ysize), sizeof(hostZ[0]));
        //hostB = (float*)calloc(size_t(Bsize), sizeof(hostB[0]));
        //hostAfterConv = (float*)calloc(size_t(Ysize), sizeof(hostAfterConv[0]));
        //hostAfterAdd = (float*)calloc(size_t(Ysize), sizeof(hostAfterAdd[0]));
        //hostAfterBias = (float*)calloc(size_t(Ysize), sizeof(hostAfterBias[0]));

        

        //for (size_t i = 0; i < Xsize; ++i)
        //{
        //    hostX[i] = static_cast<float>(image[i] - 128) / 128;
        //    //std::cout << std::setprecision(3) << hostX[i];
        //    //if (i > 1 && ((i - 1) % cols) == 0)
        //    //{
        //    //    std::cout << std::endl;
        //    //}
        //}

        //initFilter(hostW, Wsize);
        ////initFilter(hostY, Ysize);

        //checkCudaErr(cudaMemcpy(devPtrX, hostX, size_t(sizeof(hostX[0]) * Xsize), cudaMemcpyHostToDevice));
        //checkCudaErr(cudaMemcpy(devPtrW, hostW, size_t(sizeof(hostW[0]) * Wsize), cudaMemcpyHostToDevice));
        //checkCudaErr(cudaMemcpy(devPtrY, hostY, size_t(sizeof(hostY[0]) * Ysize), cudaMemcpyHostToDevice));
        //checkCudaErr(cudaMemcpy(devPtrZ, hostZ, (size_t)(sizeof(hostZ[0]) * Ysize), cudaMemcpyHostToDevice));
        //checkCudaErr(cudaMemcpy(devPtrB, hostB, (size_t)(sizeof(hostB[0]) * Bsize), cudaMemcpyHostToDevice));
        //checkCudaErr(cudaMemcpy(devPtrAfterAdd, hostAfterAdd, (size_t)(sizeof(hostAfterAdd[0]) * Ysize), cudaMemcpyHostToDevice));
        //checkCudaErr(cudaMemcpy(devPtrAfterBias, hostAfterBias, (size_t)(sizeof(hostAfterBias[0]) * Ysize), cudaMemcpyHostToDevice));
        //checkCudaErr(cudaMemcpy(devPtrAfterConv, hostAfterConv, (size_t)(sizeof(hostAfterConv[0]) * Ysize), cudaMemcpyHostToDevice));

        //checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle_;

        try
        {
            checkCudnnErr(cudnnCreate(&handle_));
            //run_serialization_conv_bias_scale_relu
            // Creates the necessary tensor descriptors
            int64_t stride[4];
            generateStrides(x_dim, stride, 4, CUDNN_TENSOR_NHWC);
            auto xTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, x_dim)
                .setStride(4, stride)
                .setId('x')
                .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                .setDataType(dataType)
                .build();
            generateStrides(w_dim, stride, 4, CUDNN_TENSOR_NHWC);
            auto wTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, w_dim)
                .setStride(4, stride)
                .setId('w')
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            generateStrides(b_dim, stride, 4, CUDNN_TENSOR_NHWC);
            auto bTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, b_dim)
                .setStride(4, stride)
                .setId('b')
                .setAlignment(16)
                .setDataType(dataType)
                .build();
            generateStrides(s_dim, stride, 4, CUDNN_TENSOR_NHWC);
            auto sTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, s_dim)
                .setStride(4, stride)
                .setId('s')
                .setAlignment(16)
                .setDataType(dataType)
                .build();

            generateStrides(y_dim, stride, 4, CUDNN_TENSOR_NHWC);
            auto afterConvTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('A')  // after conv
                .setAlignment(16)
                .setVirtual()
                .setDataType(dataType)
                .build();
            auto afterScaleTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('C')  // after scale
                .setAlignment(16)
                .setVirtual()
                .setDataType(dataType)
                .build();
            auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('B')  // after bias
                .setAlignment(16)
                .setVirtual()
                .setDataType(dataType)
                .build();
            //auto afterBiasTensor = cudnn_frontend::TensorBuilder()
            //    .setDim(4, y_dim)
            //    .setStride(4, stride)
            //    .setId('B')  // after bias
            //    .setAlignment(16)
            //    .setVirtual()
            //    .setDataType(dataType)
            //    .build();
            //auto afterScaleTensor = cudnn_frontend::TensorBuilder()
            //    .setDim(4, y_dim)
            //    .setStride(4, stride)
            //    .setId('C')  // after scale
            //    .setAlignment(16)
            //    .setVirtual()
            //    .setDataType(dataType)
            //    .build();
            auto yTensor = cudnn_frontend::TensorBuilder()
                .setDim(4, y_dim)
                .setStride(4, stride)
                .setId('y')  // output
                .setAlignment(16)
                .setDataType(CUDNN_DATA_FLOAT)
                .build();

            std::cout << xTensor.describe() << std::endl;
            std::cout << wTensor.describe() << std::endl;
            std::cout << bTensor.describe() << std::endl;
            std::cout << sTensor.describe() << std::endl;
            std::cout << afterConvTensor.describe() << std::endl;
            std::cout << afterBiasTensor.describe() << std::endl;
            std::cout << afterScaleTensor.describe() << std::endl;
            std::cout << yTensor.describe() << std::endl;

            // Define the bias descriptor
            auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_ADD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << biasDesc.describe() << std::endl;

            // Define the scale descriptor
            auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_MUL)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << scaleDesc.describe() << std::endl;

            // Define the activation descriptor
            auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                .setMode(CUDNN_POINTWISE_RELU_FWD)
                .setComputeType(CUDNN_DATA_FLOAT)
                .build();
            std::cout << actDesc.describe() << std::endl;

            // Define the convolution problem
            auto convDesc = cudnn_frontend::ConvDescBuilder()
                .setComputeType(CUDNN_DATA_FLOAT)
                .setMathMode(CUDNN_CROSS_CORRELATION)
                .setSpatialDimCount(convDim)
                .setSpatialStride(convDim, conv_strideA)
                .setPrePadding(convDim, conv_padA)
                .setPostPadding(convDim, conv_padA)
                .setDilation(convDim, conv_dilationA)
                .build();
            std::cout << convDesc.describe() << std::endl;

            float alpha = 1.0f;
            float beta = 0.0f;

            // Create a convolution Node
            auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                .setxDesc(xTensor)
                .setwDesc(wTensor)
                .setyDesc(afterConvTensor)
                .setcDesc(convDesc)
                .setAlpha(alpha)
                .setBeta(beta)
                .build();
            std::cout << conv_op.describe() << std::endl;

            // Create a Multiplication Node with scaling parameters.
            auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(conv_op.getOutputTensor())
                .setbDesc(sTensor)
                .setyDesc(afterScaleTensor)
                .setpwDesc(scaleDesc)
                .build();
            std::cout << scale_op.describe() << std::endl;

            // Create a Bias Node.
            auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(scale_op.getOutputTensor())
                .setbDesc(bTensor)
                .setyDesc(afterBiasTensor)
                .setpwDesc(biasDesc)
                .build();
            std::cout << bias_op.describe() << std::endl;

            // Create an Activation Node.
            auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(bias_op.getOutputTensor())
                .setyDesc(yTensor)
                .setpwDesc(actDesc)
                .build();
            std::cout << act_op.describe() << std::endl;

            // Create an Operation Graph. In this case it is convolution bias scale activation
            //std::array<cudnn_frontend::Operation const*, 4> ops = { &conv_op, &bias_op, &scale_op, &act_op };
            std::vector<cudnn_frontend::Operation const*> ops = { &conv_op, &scale_op, &bias_op, &act_op };
            //std::array ops = { &conv_op, &bias_op, &scale_op, &act_op };

            auto opGraph = cudnn_frontend::OperationGraphBuilder()
                .setHandle(handle_)
                .setOperationGraph(ops.size(), ops.data())
                .build();

            auto plan = get_execplan_from_heuristics_else_fall_back(std::move(opGraph), handle_);

            std::cout << "Plan tag: " << plan.getTag() << std::endl;

            auto workspace_size = plan.getWorkspaceSize();
            std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

            void* workspace_ptr = nullptr;
            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }
            void* data_ptrs[] = { devPtrX, devPtrY, devPtrW, devPtrB, devPtrS };
            int64_t uids[] = { 'x', 'y', 'w', 'b', 's' };
            auto variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                .setDataPointers(5, data_ptrs)
                .setUids(5, uids)
                .build();
            std::cout << "variantPack " << variantPack.describe() << std::endl;
            cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
            if (workspace_size > 0) {
                checkCudaErr(cudaFree(workspace_ptr));
            }

        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }

        cudaDeviceSynchronize();
        //cudaMemcpy(hostY, devPtrY, (size_t)(sizeof(hostY[0]) * Ysize), cudaMemcpyDeviceToHost);
        checkCudaErr(cudaMemcpy(Y.hostPtr, Y.devPtr, (size_t)(sizeof(Y.hostPtr[0])* Ysize), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        display(Y.hostPtr, label);
        //display(hostY, label);

        //if (devPtrX) cudaFree(devPtrX);
        //if (devPtrW) cudaFree(devPtrW);
        //if (devPtrY) cudaFree(devPtrY);
        //if (devPtrZ) cudaFree(devPtrZ);
        //if (devPtrB) cudaFree(devPtrB);
        //if (devPtrAfterAdd) cudaFree(devPtrAfterAdd);
        //if (devPtrAfterBias) cudaFree(devPtrAfterBias);
        //if (devPtrAfterConv) cudaFree(devPtrAfterConv);

        //if (hostX) free(hostX);
        //if (hostW) free(hostW);
        //if (hostY) free(hostY);
        //if (hostZ) free(hostZ);
        //if (hostB) free(hostB);
        //if (hostAfterAdd) free(hostAfterAdd);
        //if (hostAfterBias) free(hostAfterBias);
        //if (hostAfterConv) free(hostAfterConv);
    }

    export void LenetForwardv0_2()
    {
        std::cout << "Rigid LeNet Test v0.2" << std::endl;

        using namespace Helpers;

        int64_t dimA[] = { 1, 1, 28, 28 };
        int64_t filterdimA[] = { 1, 1, 5, 5 };
        int64_t outdimA[] = { 0, 0, 0, 0 }; // Computed Below
        int64_t padA[] = { 2, 2 };
        int64_t dilationA[] = { 1, 1 };
        int64_t convstrideA[] = { 1, 1 };

        int numErrors = 0;

        outdimA[0] = dimA[0];
        outdimA[1] = filterdimA[0];
        for (int dim = 0; dim < 2; dim++) {
            outdimA[dim + 2] = Helpers::getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
        }

        cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

        printf("====DIMENSIONS====\n");
        printf("input dims are %lld, %lld, %lld, %lld\n", dimA[0], dimA[1], dimA[2], dimA[3]);
        printf("filter dims are %lld, %lld, %lld, %lld\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
        printf("output dims are %lld, %lld, %lld, %lld\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


        int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
        int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
        int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];
        int64_t Bsize = outdimA[0] * outdimA[1] * 1 * 1;

        float* devPtrX = NULL;
        float* devPtrW = NULL;
        float* devPtrY = NULL;
        float* devPtrZ = NULL;
        float* devPtrB = NULL;
        float* devPtrAfterAdd = NULL;
        float* devPtrAfterConv = NULL;
        float* devPtrAfterBias = NULL;

        float* hostX = NULL;
        float* hostW = NULL;
        float* hostY = NULL;
        float* hostZ = NULL;
        float* hostB = NULL;
        float* hostAfterAdd = NULL;
        float* hostAfterConv = NULL;
        float* hostAfterBias = NULL;

        checkCudaErr(cudaMalloc((void**)&(devPtrX), size_t((Xsize) * sizeof(devPtrX[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrW), size_t((Wsize) * sizeof(devPtrW[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrY), size_t((Ysize) * sizeof(devPtrY[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrZ), size_t((Ysize) * sizeof(devPtrZ[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrB), size_t((Bsize) * sizeof(devPtrB[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrAfterConv), size_t((Ysize) * sizeof(devPtrAfterConv[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrAfterAdd), size_t((Ysize) * sizeof(devPtrAfterAdd[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrAfterBias), size_t((Ysize) * sizeof(devPtrAfterBias[0]))));

        hostX = (float*)calloc(size_t(Xsize), sizeof(hostX[0]));
        hostW = (float*)calloc(size_t(Wsize), sizeof(hostW[0]));
        hostY = (float*)calloc(size_t(Ysize), sizeof(hostY[0]));
        hostZ = (float*)calloc(size_t(Ysize), sizeof(hostZ[0]));
        hostB = (float*)calloc(size_t(Bsize), sizeof(hostB[0]));
        hostAfterConv = (float*)calloc(size_t(Ysize), sizeof(hostAfterConv[0]));
        hostAfterAdd = (float*)calloc(size_t(Ysize), sizeof(hostAfterAdd[0]));
        hostAfterBias = (float*)calloc(size_t(Ysize), sizeof(hostAfterBias[0]));

        MNISTDataHolder dh;
        dh.initialize();
        auto [image, label] = dh.getNextTrain();
        auto [rows, cols] = dh.getDimensions();

        for (int64_t i = 0; i < Xsize; ++i)
        {
            hostX[i] = static_cast<float>(image[i] - 128) / 128;
            //std::cout << std::setprecision(3) << hostX[i];
            //if (i > 1 && ((i - 1) % cols) == 0)
            //{
            //    std::cout << std::endl;
            //}
        }

        initFilter(hostW, Wsize);
        //initFilter(hostY, Ysize);

        checkCudaErr(cudaMemcpy(devPtrX, hostX, size_t(sizeof(hostX[0]) * Xsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrW, hostW, size_t(sizeof(hostW[0]) * Wsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrY, hostY, size_t(sizeof(hostY[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrZ, hostZ, (size_t)(sizeof(hostZ[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrB, hostB, (size_t)(sizeof(hostB[0]) * Bsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrAfterAdd, hostAfterAdd, (size_t)(sizeof(hostAfterAdd[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrAfterBias, hostAfterBias, (size_t)(sizeof(hostAfterBias[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrAfterConv, hostAfterConv, (size_t)(sizeof(hostAfterConv[0]) * Ysize), cudaMemcpyHostToDevice));

        checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle_;
        static cudnn_frontend::ExecutionPlanCache plan_cache("sample_cache");

        try
        {
            checkCudnnErr(cudnnCreate(&handle_));
            common_convbias_descriptors tensors = create_lenet_descriptors(
                dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT);

            std::cout << std::get<X_TENSOR>(tensors).describe() << std::endl;
            std::cout << std::get<Y_TENSOR>(tensors).describe() << std::endl;
            std::cout << std::get<W_TENSOR>(tensors).describe() << std::endl;
            std::cout << std::get<Z_TENSOR>(tensors).describe() << std::endl;
            std::cout << std::get<B_TENSOR>(tensors).describe() << std::endl;
            std::cout << std::get<AFTERADD_TENSOR>(tensors).describe() << std::endl;
            std::cout << std::get<AFTERBIAS_TENSOR>(tensors).describe() << std::endl;
            std::cout << std::get<AFTERCONV_TENSOR>(tensors).describe() << std::endl;


            auto opGraph = create_lenet_operation_graph(tensors, padA, convstrideA, dilationA, handle_);
            std::cout << opGraph.describe() << std::endl;
            //void* data_ptrs[] = { devPtrX, devPtrY, devPtrW };
            //int64_t uids[] = { 'x', 'y', 'w' };

            auto filtered_configs = generateConfigList(opGraph);
            //auto filtered_configs = heurgen_method(opGraph);

            for (auto& filtered_config : filtered_configs) {
                try {
                    auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_config, opGraph.getTag()).build();

                    //std::array<cudnn_frontend::GeneratorSource const, 1> sources = { heurgen_method };
                    //cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());
                    //auto workspace_size = 100 * 1024 * 1024; // 100 MB

                    std::cout << "Plan tag: " << plan.getTag() << std::endl;

                    auto workspace_size = plan.getWorkspaceSize();
                    std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

                    void* workspace_ptr = nullptr;
                    if (workspace_size > 0) {
                        checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
                    }
                    void* data_ptrs[] = { devPtrX, devPtrY, devPtrW, devPtrZ, devPtrB/*, devPtrAfterAdd, devPtrAfterBias, devPtrAfterConv*/ };
                    int64_t uids[] = { 'x', 'y', 'w', 'z', 'b'/*, 'A', 'B', 'C'*/ };
                    int64_t num_ptr = sizeof(uids) / sizeof(uids[0]);
                    std::cout << std::format("Num ptrs: {}", num_ptr) << std::endl;

                    auto variantPack = cudnn_frontend::VariantPackBuilder()
                        .setWorkspacePointer(workspace_ptr)
                        .setDataPointers(num_ptr, data_ptrs)
                        .setUids(num_ptr, uids)
                        .build();
                    std::cout << "variantPack " << variantPack.describe() << std::endl;

                    //auto plan = generator.cudnnFindPlanAndCache<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
                        //handle_, opGraph, variantPack, plan_cache);
                    //std::cout << "Plan tag: " << plan.getTag() << std::endl;

                    cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
                    if (workspace_size > 0) {
                        checkCudaErr(cudaFree(workspace_ptr));
                    }
                    cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
                    std::cout << "Test completed succesfully" << std::endl;
                    break;
                }
                catch (cudnn_frontend::cudnnException& e) {
                    std::cout << "Operation failed: " << e.what() << std::endl;
                    continue;
                }
            }

            //std::array<cudnn_frontend::GeneratorSource const, 1> sources = { heurgen_method };
            //cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());

            //auto workspace_size = 100 * 1024 * 1024; // 100 MB
            //void* workspace_ptr = nullptr;
            //if (workspace_size > 0) {
            //    checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            //}

            //auto variantPack = cudnn_frontend::VariantPackBuilder()
            //    .setWorkspacePointer(workspace_ptr)
            //    .setDataPointers(3, data_ptrs)
            //    .setUids(3, uids)
            //    .build();
            //std::cout << "variantPack " << variantPack.describe() << std::endl;

            //auto plan = generator.cudnnFindPlanAndCache<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
            //    handle_, opGraph, variantPack, plan_cache);

            //std::cout << "Plan tag: " << plan.getTag() << " finished in " << plan.getExecutionTime() << " ms,"
            //    << " workspace: " << plan.getWorkspaceSize() << " bytes" << std::endl;

            //cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());

            //if (workspace_size > 0) {
            //    checkCudaErr(cudaFree(workspace_ptr));
            //}
            //cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);



        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }

        cudaDeviceSynchronize();
        cudaMemcpy(hostY, devPtrY, (size_t)(sizeof(hostY[0]) * Ysize), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();


        display(hostY, label);

        if (devPtrX) cudaFree(devPtrX);
        if (devPtrW) cudaFree(devPtrW);
        if (devPtrY) cudaFree(devPtrY);
        if (devPtrZ) cudaFree(devPtrZ);
        if (devPtrB) cudaFree(devPtrB);
        if (devPtrAfterAdd) cudaFree(devPtrAfterAdd);
        if (devPtrAfterBias) cudaFree(devPtrAfterBias);
        if (devPtrAfterConv) cudaFree(devPtrAfterConv);

        if (hostX) free(hostX);
        if (hostW) free(hostW);
        if (hostY) free(hostY);
        if (hostZ) free(hostZ);
        if (hostB) free(hostB);
        if (hostAfterAdd) free(hostAfterAdd);
        if (hostAfterBias) free(hostAfterBias);
        if (hostAfterConv) free(hostAfterConv);
    }

    export void RigidTest()
    {
        std::cout << "Rigid LeNet Test v0.1" << std::endl;

        using namespace Helpers;

        int64_t dimA[] = { 1, 1, 28, 28 };
        int64_t filterdimA[] = { 6, 1, 5, 5 };
        int64_t outdimA[] = { 0, 0, 0, 0 }; // Computed Below
        int64_t padA[] = { 2, 2 };
        int64_t dilationA[] = { 1, 1 };
        int64_t convstrideA[] = { 1, 1 };

        int numErrors = 0;

        Helpers::bark();

        outdimA[0] = dimA[0];
        outdimA[1] = filterdimA[0];
        for (int dim = 0; dim < 2; dim++) {
            outdimA[dim + 2] = Helpers::getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
        }


        cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

        printf("====DIMENSIONS====\n");
        printf("input dims are %lld, %lld, %lld, %lld\n", dimA[0], dimA[1], dimA[2], dimA[3]);
        printf("filter dims are %lld, %lld, %lld, %lld\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
        printf("output dims are %lld, %lld, %lld, %lld\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


        int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
        int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
        int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

        float* devPtrX = NULL;
        float* devPtrW = NULL;
        float* devPtrY = NULL;

        float* hostX = NULL;
        float* hostW = NULL;
        float* hostY = NULL;

        checkCudaErr(cudaMalloc((void**)&(devPtrX), size_t((Xsize) * sizeof(devPtrX[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrW), size_t((Wsize) * sizeof(devPtrW[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrY), size_t((Ysize) * sizeof(devPtrY[0]))));

        hostX = (float*)calloc(size_t(Xsize), sizeof(hostX[0]));
        hostW = (float*)calloc(size_t(Wsize), sizeof(hostW[0]));
        hostY = (float*)calloc(size_t(Ysize), sizeof(hostY[0]));

        MNISTDataHolder dh;
        dh.initialize();
        auto [image, label] = dh.getNextTrain();
        auto [rows, cols] = dh.getDimensions();
        //std::vector<float> input_data(rows * cols);

        for (int64_t i = 0; i < Xsize; ++i)
        {
            hostX[i] = static_cast<float>(image[i] - 128) / 128;
            std::cout << hostX[i];
            if (i > 0 && i % cols == 0)
            {
                std::cout << std::endl;
            }
        }

        initFilter(hostW, Wsize);

        checkCudaErr(cudaMemcpy(devPtrX, hostX, size_t(sizeof(hostX[0]) * Xsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrW, hostW, size_t(sizeof(hostW[0]) * Wsize), cudaMemcpyHostToDevice));
        //checkCudaErr(cudaMemcpy(devPtrY, hostY, size_t(sizeof(hostY[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());

        cudnnHandle_t handle_;
        static cudnn_frontend::ExecutionPlanCache plan_cache("sample_cache");

        try
        {
            checkCudnnErr(cudnnCreate(&handle_));
            //common_conv_descriptors descriptors = create_common_descriptors(
            //    dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, mode);

            //std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
            //std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
            //std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
            //std::cout << std::get<3>(descriptors).describe() << std::endl;

            //auto opGraph = create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, handle_);
            //std::cout << opGraph.describe() << std::endl;
            //void* data_ptrs[] = { devPtrX, devPtrY, devPtrW };
            //int64_t uids[] = { 'x', 'y', 'w' };

            //std::array<cudnn_frontend::GeneratorSource const, 1> sources = { heurgen_method };
            //cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());

            //auto workspace_size = 100 * 1024 * 1024; // 100 MB
            //void* workspace_ptr = nullptr;
            //if (workspace_size > 0) {
            //    checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            //}

            //auto variantPack = cudnn_frontend::VariantPackBuilder()
            //    .setWorkspacePointer(workspace_ptr)
            //    .setDataPointers(3, data_ptrs)
            //    .setUids(3, uids)
            //    .build();
            //std::cout << "variantPack " << variantPack.describe() << std::endl;

            //auto plan = generator.cudnnFindPlanAndCache<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
            //    handle_, opGraph, variantPack, plan_cache);

            //std::cout << "Plan tag: " << plan.getTag() << " finished in " << plan.getExecutionTime() << " ms,"
            //    << " workspace: " << plan.getWorkspaceSize() << " bytes" << std::endl;

            //cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());

            //if (workspace_size > 0) {
            //    checkCudaErr(cudaFree(workspace_ptr));
            //}
            //cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
        }
        catch (cudnn_frontend::cudnnException& e) {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
        }

        cudaDeviceSynchronize();
        cudaMemcpy(hostY, devPtrY, (size_t)(sizeof(hostY[0]) * Ysize), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();


        display(hostY, label);

        if (handle_) cudnnDestroy(handle_);
        if (devPtrX) cudaFree(devPtrX);
        if (devPtrW) cudaFree(devPtrW);
        if (devPtrY) cudaFree(devPtrY);

        if (hostX) free(hostX);
        if (hostW) free(hostW);
        if (hostY) free(hostY);
    }
}
