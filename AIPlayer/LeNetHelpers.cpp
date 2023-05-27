#include "LeNetHelpers.h"

#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>

import <iostream>;
import <array>;
import <vector>;
import <random>;
import <numeric>;

namespace Helpers
{
    void bark()
    {
        std::cout << "Bark" << std::endl;
    }


    int64_t checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
        if (code) {
            printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int)code, cudaGetErrorString(code), expr);
            return 1;
        }
        return 0;
    }

    int64_t checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
        if (code) {
            printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int)code, cudnnGetErrorString(code), expr);
            return 1;
        }
        return 0;
    }

    bool
        allowAll(cudnnBackendDescriptor_t engine_config) {
        (void)engine_config;
        return false;
    }

    bool
        isNonDeterministic(cudnnBackendDescriptor_t engine_config) {
        return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(engine_config);
    }

    void generateStrides(const int64_t* dimA, int64_t* strideA, int64_t nbDims, cudnnTensorFormat_t filterFormat) {
        // For INT8x4 and INT8x32 we still compute standard strides here to input
        // into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
        if (filterFormat == CUDNN_TENSOR_NCHW) {
            strideA[nbDims - 1] = 1;
            for (int64_t d = nbDims - 2; d >= 0; d--) {
                strideA[d] = strideA[d + 1] * dimA[d + 1];
            }
        }
        else {
            // Here we assume that the format is CUDNN_TENSOR_NHWC
            strideA[1] = 1;
            strideA[nbDims - 1] = strideA[1] * dimA[1];
            for (int64_t d = nbDims - 2; d >= 2; d--) {
                strideA[d] = strideA[d + 1] * dimA[d + 1];
            }
            strideA[0] = strideA[2] * dimA[2];
        }
    }

    common_conv_descriptors
    create_common_descriptors(int64_t* x_dim,
        int64_t* padA,
        int64_t* convstrideA,
        int64_t* dilationA,
        int64_t* w_dim,
        int64_t* y_dim,
        cudnnDataType_t dataType,
        cudnnConvolutionMode_t mode) {
        const int convDim = 2;
    
        int64_t strideA[4];
        int64_t outstrideA[4];
        int64_t filterstrideA[4];
    
        generateStrides(w_dim, filterstrideA, 4, CUDNN_TENSOR_NCHW);
        generateStrides(x_dim, strideA, 4, CUDNN_TENSOR_NCHW);
        generateStrides(y_dim, outstrideA, 4, CUDNN_TENSOR_NCHW);
    
        return common_conv_descriptors(cudnn_frontend::TensorBuilder()
            .setDim(4, x_dim)
            .setStride(4, strideA)
            .setId('x')
            .setAlignment(4)
            .setDataType(dataType)
            .build(),
            cudnn_frontend::TensorBuilder()
            .setDim(4, y_dim)
            .setStride(4, outstrideA)
            .setId('y')
            .setAlignment(4)
            .setDataType(dataType)
            .build(),
            cudnn_frontend::TensorBuilder()
            .setDim(4, w_dim)
            .setStride(4, filterstrideA)
            .setId('w')
            .setAlignment(4)
            .setDataType(dataType)
            .build(),
            cudnn_frontend::ConvDescBuilder()
            .setComputeType(dataType)
            .setMathMode(mode)
            .setSpatialDimCount(convDim)
            .setSpatialStride(convDim, convstrideA)
            .setPrePadding(convDim, padA)
            .setPostPadding(convDim, padA)
            .setDilation(convDim, dilationA)
            .build());
    }

    common_convbias_descriptors create_lenet_descriptors(
            int64_t* x_dim,
            int64_t* padA,
            int64_t* convstrideA,
            int64_t* dilationA,
            int64_t* w_dim,
            int64_t* y_dim,
            cudnnDataType_t dataType,
            cudnnDataType_t computeType) {
        (void)padA;
        (void)convstrideA;
        (void)dilationA;
        int64_t b_dim[4];
        b_dim[0] = 1;
        b_dim[1] = y_dim[1];
        b_dim[2] = 1;
        b_dim[3] = 1;

        int64_t x_stride[4];
        int64_t y_stride[4];
        int64_t w_stride[4];
        int64_t b_stride[4];

        generateStrides(w_dim, w_stride, 4, CUDNN_TENSOR_NHWC);
        generateStrides(x_dim, x_stride, 4, CUDNN_TENSOR_NHWC);
        generateStrides(y_dim, y_stride, 4, CUDNN_TENSOR_NHWC);
        generateStrides(b_dim, b_stride, 4, CUDNN_TENSOR_NHWC);

        return common_convbias_descriptors(
            cudnn_frontend::TensorBuilder()
            .setDim(4, x_dim)
            .setStride(4, x_stride)
            .setId('x')
            .setAlignment(4)
            .setDataType(dataType)
            .build(),
            cudnn_frontend::TensorBuilder()
            .setDim(4, y_dim)
            .setStride(4, y_stride)
            .setId('y')
            .setAlignment(4)
            .setDataType(dataType)
            .build(),
            cudnn_frontend::TensorBuilder()
            .setDim(4, w_dim)
            .setStride(4, w_stride)
            .setId('w')
            .setAlignment(4)
            .setDataType(dataType)
            .build(),
            cudnn_frontend::TensorBuilder()
            .setDim(4, y_dim)
            .setStride(4, y_stride)
            .setId('z')
            .setAlignment(4)
            .setDataType(dataType)
            .build(),
            cudnn_frontend::TensorBuilder()
            .setDim(4, b_dim)
            .setStride(4, b_stride)
            .setId('b')
            .setAlignment(4)
            .setDataType(dataType)
            .build(),
            cudnn_frontend::TensorBuilder()
            .setDim(4, y_dim)
            .setStride(4, y_stride)
            .setVirtual()
            .setId('A')  // after add
            .setAlignment(4)
            .setDataType(computeType)
            .build(),
            cudnn_frontend::TensorBuilder()
            .setDim(4, y_dim)
            .setStride(4, y_stride)
            .setVirtual()
            .setId('B')  // after bias
            .setAlignment(4)
            .setDataType(computeType)
            .build(),
            cudnn_frontend::TensorBuilder()
            .setDim(4, y_dim)
            .setStride(4, y_stride)
            .setId('C')  // after conv
            .setAlignment(4)
            .setVirtual()
            .setDataType(computeType)
            .build()
        );
    }

    cudnn_frontend::OperationGraph
    create_operation_graph(
        common_conv_descriptors& descriptors, 
        cudnnBackendDescriptorType_t mode, //CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
        cudnnHandle_t handle_) 
    {
        float alpha = 1.0f;
        float beta = 0.0;
    
        auto conv_op = cudnn_frontend::OperationBuilder(mode)
            .setxDesc(std::get<X_TENSOR>(descriptors))
            .setyDesc(std::get<Y_TENSOR>(descriptors))
            .setwDesc(std::get<W_TENSOR>(descriptors))
            .setcDesc(std::get<3>(descriptors))
            .setAlpha(alpha)
            .setBeta(beta)
            .build();
    
        std::cout << "Operation is " << conv_op.describe() << std::endl;
        std::array<cudnn_frontend::Operation const*, 1> ops = { &conv_op};
    
        return cudnn_frontend::OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();
    }

    cudnn_frontend::OperationGraph create_lenet_operation_graph(
            common_convbias_descriptors& tensors,
            int64_t* pad,
            int64_t* convstride,
            int64_t* dilation,
            cudnnHandle_t handle_)
    {
        constexpr int convDim = 2;
        // Define the add operation
        auto addDesc = cudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_ADD)
            .setMathPrecision(CUDNN_DATA_FLOAT)
            .build();
        std::cout << addDesc.describe() << std::endl;

        // Define the bias operation
        auto addDesc2 = cudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_ADD)
            .setMathPrecision(CUDNN_DATA_FLOAT)
            .build();
        std::cout << addDesc2.describe() << std::endl;

        // Define the activation operation
        auto actDesc = cudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_RELU_FWD)
            .setMathPrecision(CUDNN_DATA_FLOAT)
            .build();
        std::cout << actDesc.describe() << std::endl;

        // Define the convolution problem
        auto convDesc = cudnn_frontend::ConvDescBuilder()
            .setComputeType(CUDNN_DATA_FLOAT)
            .setMathMode(CUDNN_CONVOLUTION)
            .setSpatialDimCount(convDim)
            .setSpatialStride(convDim, convstride)
            .setPrePadding(convDim, pad)
            .setPostPadding(convDim, pad)
            .setDilation(convDim, dilation)
            .build();
        std::cout << convDesc.describe() << std::endl;

        float alpha = 1.0f;
        float alpha2 = 0.5f;
        float beta = 0.0f;

        // Create a convolution Node
        auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            .setxDesc(std::get<X_TENSOR>(tensors))
            .setwDesc(std::get<W_TENSOR>(tensors))
            .setyDesc(std::get<AFTERCONV_TENSOR>(tensors))
            .setcDesc(convDesc)
            .setAlpha(alpha)
            .setBeta(beta)
            .build();
        std::cout << conv_op.describe() << std::endl;

        // Create a Add Node with scaling parameters.
        auto add_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(conv_op.getOutputTensor())
            .setbDesc(std::get<Z_TENSOR>(tensors))
            .setyDesc(std::get<AFTERADD_TENSOR>(tensors))
            .setpwDesc(addDesc)
            .setAlpha(alpha)
            .setAlpha2(alpha2)
            .build();
        std::cout << add_op1.describe() << std::endl;

        // Create a Bias Node.
        auto add_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(add_op1.getOutputTensor())
            .setbDesc(std::get<B_TENSOR>(tensors))
            .setyDesc(std::get<AFTERBIAS_TENSOR>(tensors))
            .setpwDesc(addDesc2)
            .build();
        std::cout << add_op2.describe() << std::endl;

        //auto add_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
        //    .setxDesc(conv_op.getOutputTensor())
        //    .setbDesc(std::get<B_TENSOR>(tensors))
        //    .setyDesc(std::get<AFTERBIAS_TENSOR>(tensors))
        //    .setpwDesc(addDesc2)
        //    .build();
        //std::cout << add_op2.describe() << std::endl;

        // Create an Activation Node.
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
            .setxDesc(add_op2.getOutputTensor())
            .setyDesc(std::get<Y_TENSOR>(tensors))
            .setpwDesc(actDesc)
            .build();
        std::cout << act_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is convolution add bias activation
        std::array<cudnn_frontend::Operation const*, 4> ops = { &conv_op, &add_op1, &add_op2, &act_op };
        //std::array<cudnn_frontend::Operation const*, 3> ops = { &conv_op, &add_op2, &act_op };

        auto opGraph = cudnn_frontend::OperationGraphBuilder()
            .setHandle(handle_)
            .setOperationGraph(ops.size(), ops.data())
            .build();

        return opGraph;
    }


    int64_t getFwdConvDilatedFilterDim(int64_t filterDim, int64_t dilation) {
        return ((filterDim - 1) * dilation) + 1;
    }

    int64_t getFwdConvPaddedImageDim(int64_t tensorDim, int64_t pad) {
        return tensorDim + (2 * pad);
    }

    int64_t getFwdConvOutputDim(
        int64_t tensorDim,
        int64_t pad,
        int64_t filterDim,
        int64_t stride,
        int64_t dilation)
    {
        int64_t p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
        return (p);
    }

    void initFilter(float* hostW, size_t size)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t i = 0; i < size; ++i)
        {
            hostW[i] = dist(rng);
            //hostW[i] = 1.0f;
        }
    }

    cudnn_frontend::EngineConfigList generateConfigList(cudnn_frontend::OperationGraph& opGraph)
    {
        // How many engines support this operation graph ?
        auto total_engines = opGraph.getEngineCount();
        std::cout << "conv__activation " << opGraph.describe() << " has " << total_engines << " engines." << std::endl;

        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses =
            cudnn_frontend::get_heuristics_list<2>({ "heuristics_instant"
                , "heuristics_fallback"
                }, opGraph, isNonDeterministic, filtered_configs);

        std::cout << "get_heuristics_list Statuses: ";
        for (size_t i = 0; i < statuses.size(); i++) {
            std::cout << cudnn_frontend::to_string(statuses[i]) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;
        return filtered_configs;
    }

    cudnn_frontend::ExecutionPlan get_execplan_from_heuristics_else_fall_back(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_) {
#if (CUDNN_VERSION >= 8200)
            {
                auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                    .setOperationGraph(opGraph)
                    .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                    .build();

                std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
                auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

                // Try engine configs returned by the heuristics and pick up the first one that works.
                for (auto& ecfg : engine_config) {
                    try {
                        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                            .setHandle(handle_)
                            .setEngineConfig(ecfg, opGraph.getTag())
                            .build();
                        return plan;
                    }
                    catch (cudnn_frontend::cudnnException& e) {
                        continue;
                    }
                }
            }
#endif

            {
                cudnn_frontend::EngineConfigList filtered_configs;
                auto statuses =
                    cudnn_frontend::get_heuristics_list<1>({
                    "heuristics_fallback"
                        }, opGraph, allowAll, filtered_configs, true);

                std::cout << "get_heuristics_list Statuses: ";
                for (auto status : statuses) {
                    std::cout << cudnn_frontend::to_string(status) << " ";
                }
                std::cout << std::endl;
                std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

                return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
            }
    }

    // Generate uniform numbers [0,1)
    void initImage(float* image, int64_t imageSize) {
        initImagev2(image, imageSize);
        return;

        static unsigned seed = 123456789;
        for (int64_t index = 0; index < imageSize; index++) {
            seed = (1103515245 * seed + 12345) & 0xffffffff;
            image[index] = float(seed) * 2.3283064e-10f;  // 2^-32
            //image[index] = float(seed) * std::numeric_limits<float>::max();  // 2^-32
        }
    }
    // Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
    void initImage(int8_t* image, int64_t imageSize) {
        static unsigned seed = 123456789;
        for (int64_t index = 0; index < imageSize; index++) {
            seed = (1103515245 * seed + 12345) & 0xffffffff;
            // Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
            image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10f);  // 2^-32
        }
    }

    void initImagev2(float* image, int64_t imageSize)
    {
        std::random_device dev;
        std::mt19937 gen(dev());
        //std::default_random_engine generator;
        float deviation = sqrt(2.0f / imageSize);
        std::uniform_real_distribution<float> distribution(-deviation, deviation);
        //std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int64_t i = 0; i < imageSize; ++i)
        {
            image[i] = distribution(gen);
            //image [i] = distribution(generator);
            //hostW[i] = 1.0f;
        }
    }
}