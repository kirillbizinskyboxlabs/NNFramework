#pragma once


//#include "../deps/cudnn-frontend/samples/helpers.h"
//#include "../deps/cudnn-frontend/samples/conv_sample.h"
//#include "../deps/cudnn-frontend/samples/cpu_references.h"

//#define _CRT_SECURE_NO_WARNINGS 1

#include "SimpleFrontendTest.h"

#include <cudnn_frontend.h>
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>
#include <cublas.h>
//#include <cudnn.h>

//export module SimpleFrontendTest;

//import <iostream>;
import <vector>;
import <assert.h>;
import <utility>;
import <array>;

//#include "../deps/cudnn-frontend/include/cudnn_frontend.h"
//#include "../deps/cudnn-frontend/include/cudnn_frontend_find_plan.h"
//#include "../deps/cudnn-frontend/include/cudnn_frontend_get_plan.h"
//#include "../deps/cudnn-frontend/include/cudnn_frontend_EngineConfigGenerator.h"



//export void Test()
//{
//    // Create a CUDNN frontend context
//    //cudnn_frontend::Context context = cudnn_frontend::createContext();
//    cudnnHandle_t cudnn_handle;
//    cudnnCreate(&cudnn_handle);
//
//    // Define the input tensor shape
//    std::vector<int64_t> input_shape = { 1, 3, 32, 32 };
//
//    // Define the convolution kernel shape and stride
//    std::vector<int64_t> kernel_shape = { 16, 3, 5, 5 };
//    std::vector<int64_t> stride = { 1, 1, 1, 1 };
//
//    // Define the convolution operation
//    cudnn_frontend::Operation conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
//        .set("input", cudnn_frontend::TensorBuilder()
//            .set("shape", input_shape)
//            .set("data_type", cudnn_frontend::DataType::FLOAT))
//        .set("filter", cudnn_frontend::TensorBuilder()
//            .set("shape", kernel_shape)
//            .set("data_type", cudnn_frontend::DataType::FLOAT))
//        .set("stride", stride)
//        .set("padding", std::vector<int64_t>{2, 2})
//        .build();
//
//    // Define the output tensor shape
//    std::vector<int64_t> output_shape = conv_op.getOutputTensorShape("output");
//
//    // Create the output tensor
//    cudnn_frontend::Tensor output_tensor = cudnn_frontend::TensorBuilder()
//        .set("shape", output_shape)
//        .set("data_type", cudnn_frontend::DataType::FLOAT)
//        .build();
//
//    // Add the convolution operation to the context
//    context.operationGraph().addOperation(conv_op);
//
//    // Allocate memory for the input and output tensors
//    size_t input_bytes = cudnn_frontend::getTensorSizeInBytes(input_shape, cudnn_frontend::DataType::FLOAT);
//    size_t output_bytes = cudnn_frontend::getTensorSizeInBytes(output_shape, cudnn_frontend::DataType::FLOAT);
//    void* input_data = malloc(input_bytes);
//    void* output_data = malloc(output_bytes);
//
//    // Initialize the input tensor data (omitted for brevity)
//
//    // Run the operation
//    context.execute(input_data, output_data);
//
//    // Print the output tensor data (omitted for brevity)
//
//    // Free the input and output memory
//    free(input_data);
//    free(output_data);
//
//    cudnnDestroy(cudnn_handle);
//}

//void run_from_heuristics(
//    int64_t* dimA_padded,
//    int64_t* padA,
//    int64_t* convstrideA,
//    int64_t* dilationA,
//    int64_t* filterdimA_padded,
//    int64_t* outdimA_padded,
//    cudnnDataType_t dataType,
//    cudnnConvolutionMode_t mode,
//    float* devPtrI,
//    float* devPtrF,
//    float* devPtrO,
//    cudnnBackendHeurMode_t heur_mode,
//    bool expect_in_cache = false);

//int64_t getFwdConvDilatedFilterDim(int64_t filterDim, int64_t dilation) {
//    return ((filterDim - 1) * dilation) + 1;
//}
//
//int64_t getFwdConvPaddedImageDim(int64_t tensorDim, int64_t pad) {
//    return tensorDim + (2 * pad);
//}
//
//int64_t getFwdConvOutputDim(
//    int64_t tensorDim,
//    int64_t pad,
//    int64_t filterDim,
//    int64_t stride,
//    int64_t dilation)
//{
//    int64_t p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
//    return (p);
//}

#define THRESHOLD 2.0e-2
//typedef __half half1;


//template <typename T_ELEM>
//class SurfaceManager;

// T_ELEM is the type the data is stored in, T_MATH is the type the calculations are done in.
template <typename T_ELEM, typename T_MATH>
void conv_cpu_ref(
    const T_ELEM* inputData,
    const T_ELEM* filterData,
    T_ELEM* outputData,
    int resizeFactor,
    cudnnTensorFormat_t filterFormat,
    const int64_t* inDims,
    const int64_t* filDims,
    const int64_t* diffDims,
    const int64_t* stride,
    const int64_t* pad,
    const int64_t* dilation,
    int64_t nbDims);

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

// Generate uniform numbers [0,1)
void initImage(float* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        image[index] = float(seed) * 2.3283064e-10f;  // 2^-32
    }
}

//void testinitImage(half1* image, int64_t imageSize, int test) {
//    static unsigned seed = 123456789;
//    for (int64_t index = 0; index < imageSize; index++) {
//        seed = (1103515245 * seed + 12345) & 0xffffffff;
//        // image[index] = cpu_float2half_rn(float(seed) * 2.3283064e-10f);  // 2^-32
//        if (test) image[index] = cpu_float2half_rn(static_cast<float>((index + 1) * 2));  // 2^-32
//        else image[index] = cpu_float2half_rn(static_cast<float>(index + 1));  // 2^-32
//
//    }
//}
//
//void initImage(half1* image, int64_t imageSize) {
//    static unsigned seed = 123456789;
//    for (int64_t index = 0; index < imageSize; index++) {
//        seed = (1103515245 * seed + 12345) & 0xffffffff;
//        image[index] = cpu_float2half_rn(float(seed) * 2.3283064e-10f);  // 2^-32
//    }
//}

// Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
void initImage(int8_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
        image[index] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate random integers [0, 50] to avoid uint8 overflow
void initImage(uint8_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 50]
        image[index] = (uint8_t)(50 * float(seed) * 2.3283064e-10f);  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
void initImage(int32_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int32_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32
    }
}

// Currently set to generate uniform integers [0,1]
void initImage(int64_t* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        image[index] = ((int64_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32
    }
}

// Currently set to generate booleans
void initImage(bool* image, int64_t imageSize) {
    static unsigned seed = 123456789;
    for (int64_t index = 0; index < imageSize; index++) {
        seed = (1103515245 * seed + 12345) & 0xffffffff;
        // Takes floats from [0, 1), scales and casts to ints from [0, 4], then divides by 4
        int64_t val = ((int32_t)(5.f * float(seed) * 2.3283064e-10f)) / 4;  // 2^-32

        // val is 0 or 1
        image[index] = (val == 1);
    }
}

void initImagePadded(int8_t* image, int64_t dimA[], int64_t dimPadded[], int64_t stridePadded[], cudnnDataType_t dataType) {
    static unsigned seed = 123456789;
    int64_t resizeFactor = (dataType == CUDNN_DATA_INT8x4) ? 4 : 32;
    int64_t totalSize = dimPadded[0] * dimPadded[1] * dimPadded[2] * dimPadded[3];

    // #pragma omp parallel for
    for (int64_t i = 0; i < totalSize; i++) {
        int64_t n = (i / stridePadded[0]) % dimPadded[0];
        int64_t c1 = (i / (stridePadded[1] * resizeFactor)) % (dimPadded[1] / resizeFactor);
        int64_t c2 = i % resizeFactor;
        int64_t c = c1 * resizeFactor + c2;
        if (n < dimA[0] && c < dimA[1]) {
            image[i] = 2 - (int8_t)(5 * float(seed) * 2.3283064e-10);  // 2^-32
        }
        else {
            image[i] = 0;
        }
    }
}

#define checkCudaErr(...)                                                        \
    do {                                                                         \
        int64_t err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        assert(err == 0);                                                       \
    } while (0)

#define checkCudnnErr(...)                                                        \
    do {                                                                          \
        int64_t err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); \
        assert(err == 0);                                                        \
    } while (0)

static float doFma(float fval, float ival, float tmp) {
    return fval * ival + tmp;
}

//static float doFma(half1 fval, half1 ival, float tmp) {
//    return cpu_half2float(fval) * cpu_half2float(ival) + tmp;
//}

static int32_t doFma(int8_t fval, int8_t ival, int32_t tmp) {
    return int32_t(fval) * int32_t(ival) + tmp;
}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static int32_t doFma(float fval, float ival, int32_t tmp) {
    (void)fval;
    (void)ival;
    (void)tmp;
    return 0;
}

//// Garbage function, resolves overloaded function ambiguity for an invalid type combination
//static int32_t doFma(half1 fval, half1 ival, int32_t tmp) {
//    (void)fval;
//    (void)ival;
//    (void)tmp;
//    return 0;
//}

// Garbage function, resolves overloaded function ambiguity for an invalid type combination
static float doFma(int8_t fval, int8_t ival, float tmp) {
    (void)fval;
    (void)ival;
    (void)tmp;
    return 0;
}

bool
isNonDeterministic(cudnnBackendDescriptor_t engine_config) {
    return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(engine_config);
}

bool
isReducedPrecisionReduction(cudnnBackendDescriptor_t engine_config) {
    return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION>(engine_config);
}

bool
isDownConvertingInputs(cudnnBackendDescriptor_t engine_config) {
    return cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(engine_config);
}

bool
isNonDeterministicOrisDownConverting(cudnnBackendDescriptor_t engine_config) {
    return isNonDeterministic(engine_config) || isDownConvertingInputs(engine_config);
}

bool
allowAll(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

bool allowErrata(int64_t* padA) {
    return std::all_of(padA, padA + 2, [](int64_t pad) {
        return pad == 0; });
}

bool isInt8Errata(cudnnDataType_t type) {
    return type == CUDNN_DATA_INT8;
}

enum {
    X_TENSOR,
    Y_TENSOR,
    W_TENSOR,
    Z_TENSOR,
    B_TENSOR,
    AFTERADD_TENSOR,
    AFTERBIAS_TENSOR,
    AFTERCONV_TENSOR,
};

using common_conv_descriptors =
std::tuple<cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::Tensor, cudnn_frontend::ConvDesc>;

using common_convbias_descriptors = std::tuple<cudnn_frontend::Tensor,
    cudnn_frontend::Tensor,
    cudnn_frontend::Tensor,
    cudnn_frontend::Tensor,
    cudnn_frontend::Tensor,
    cudnn_frontend::Tensor,
    cudnn_frontend::Tensor,
    cudnn_frontend::Tensor>;


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

cudnn_frontend::OperationGraph
create_operation_graph(common_conv_descriptors& descriptors, cudnnBackendDescriptorType_t mode, cudnnHandle_t handle_) {
    float alpha = 1.0f;
    float beta = 0.0;

    auto op = cudnn_frontend::OperationBuilder(mode)
        .setxDesc(std::get<X_TENSOR>(descriptors))
        .setyDesc(std::get<Y_TENSOR>(descriptors))
        .setwDesc(std::get<W_TENSOR>(descriptors))
        .setcDesc(std::get<3>(descriptors))
        .setAlpha(alpha)
        .setBeta(beta)
        .build();

    std::cout << "Operation is " << op.describe() << std::endl;

    std::array<cudnn_frontend::Operation const*, 1> ops = { &op };

    return cudnn_frontend::OperationGraphBuilder().setHandle(handle_).setOperationGraph(ops.size(), ops.data()).build();
}

// Method for engine config generator based on heuristics
auto heurgen_method = [](cudnn_frontend::OperationGraph& opGraph) -> cudnn_frontend::EngineConfigList {
    auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
        .setOperationGraph(opGraph)
        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
        .build();
    std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;

    auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    cudnn_frontend::EngineConfigList filtered_configs;
    cudnn_frontend::filter(engine_configs, filtered_configs, ::allowAll);
    return filtered_configs;
};

// Method for engine config generator based on fallback list
auto fallback_method = [](cudnn_frontend::OperationGraph& opGraph) -> cudnn_frontend::EngineConfigList {
    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
        .setOperationGraph(opGraph)
        .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        .build();
    auto& fallback_list = fallback.getFallbackList();

    cudnn_frontend::EngineConfigList filtered_configs;
    // We create this filter to pre-remove configs being passed to cudnnFind.
    // This is just a sample and is not necessary
    cudnn_frontend::filter(fallback_list, filtered_configs, ::isNonDeterministic);

    return filtered_configs;
};

void
run_from_heuristics(int64_t* x_dim,
    int64_t* padA,
    int64_t* convstrideA,
    int64_t* dilationA,
    int64_t* w_dim,
    int64_t* y_dim,
    cudnnDataType_t dataType,
    cudnnConvolutionMode_t mode,
    float* devPtrX,
    float* devPtrW,
    float* devPtrY,
    cudnnBackendHeurMode_t heur_mode,
    bool expect_in_cache = false) {
    cudnnHandle_t handle_;
    (void)heur_mode;
    static cudnn_frontend::ExecutionPlanCache plan_cache("sample_cache");
    try {
        checkCudnnErr(cudnnCreate(&handle_));
        common_conv_descriptors descriptors = create_common_descriptors(
            x_dim, padA, convstrideA, dilationA, w_dim, y_dim, dataType, mode);

        std::cout << std::get<X_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<Y_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<W_TENSOR>(descriptors).describe() << std::endl;
        std::cout << std::get<3>(descriptors).describe() << std::endl;

        auto opGraph =
            create_operation_graph(descriptors, CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, handle_);
        std::cout << opGraph.describe() << std::endl;
        void* data_ptrs[] = { devPtrX, devPtrY, devPtrW };
        int64_t uids[] = { 'x', 'y', 'w' };

        const cudnn_frontend::ExecutionPlan* cached_plan;
        if (plan_cache.get_plan_from_cache(opGraph, cached_plan)) {
            std::cout << "Cached execution plan found." << cached_plan->getTag() << std::endl;
            auto workspace_size = cached_plan->getWorkspaceSize();
            std::cout << cached_plan->describe() << " requires workspace " << workspace_size << std::endl;
            void* workspace_ptr = nullptr;
            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }
            auto variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                .setDataPointers(3, data_ptrs)
                .setUids(3, uids)
                .build();
            std::cout << "variantPack " << variantPack.describe() << std::endl;
            cudnnStatus_t status = cudnnBackendExecute(handle_, cached_plan->get_raw_desc(), variantPack.get_raw_desc());

            if (workspace_size > 0) {
                checkCudaErr(cudaFree(workspace_ptr));
            }
            cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
        }
        else {
            //REQUIRE(false == expect_in_cache);
            std::array<cudnn_frontend::GeneratorSource const, 1> sources = { heurgen_method };
            cudnn_frontend::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());

            auto workspace_size = 100 * 1024 * 1024; // 100 MB
            void* workspace_ptr = nullptr;
            if (workspace_size > 0) {
                checkCudaErr(cudaMalloc(&workspace_ptr, (size_t)workspace_size));
            }

            auto variantPack = cudnn_frontend::VariantPackBuilder()
                .setWorkspacePointer(workspace_ptr)
                .setDataPointers(3, data_ptrs)
                .setUids(3, uids)
                .build();
            std::cout << "variantPack " << variantPack.describe() << std::endl;

            auto plan = generator.cudnnFindPlanAndCache<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(
                handle_, opGraph, variantPack, plan_cache);

            std::cout << "Plan tag: " << plan.getTag() << " finished in " << plan.getExecutionTime() << " ms,"
                << " workspace: " << plan.getWorkspaceSize() << " bytes" << std::endl;

            cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());

            if (workspace_size > 0) {
                checkCudaErr(cudaFree(workspace_ptr));
            }
            cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);
        }
    }
    catch (cudnn_frontend::cudnnException& e) {
        std::cout << "[ERROR] Exception " << e.what() << std::endl;
        //CHECK(false);
    }

    if (handle_) cudnnDestroy(handle_);
    return;
}


template <typename T_ELEM>
class SurfaceManager {

public:
    T_ELEM* devPtrX = NULL;
    T_ELEM* devPtrW = NULL;
    T_ELEM* devPtrY = NULL;
    T_ELEM* devPtrZ = NULL;
    T_ELEM* devPtrB = NULL;
    T_ELEM* devPtrAfterAdd = NULL;
    T_ELEM* devPtrAfterConv = NULL;
    T_ELEM* devPtrAfterBias = NULL;

    T_ELEM* hostX = NULL;
    T_ELEM* hostW = NULL;
    T_ELEM* hostY = NULL;
    T_ELEM* hostZ = NULL;
    T_ELEM* hostB = NULL;
    T_ELEM* hostAfterAdd = NULL;
    T_ELEM* hostAfterConv = NULL;
    T_ELEM* hostAfterBias = NULL;
    T_ELEM* host_ref = NULL;


    explicit SurfaceManager(int64_t Xsize, int64_t Wsize, int64_t Ysize, int64_t ref_size) {
        checkCudaErr(cudaMalloc((void**)&(devPtrX), size_t((Xsize) * sizeof(devPtrX[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrW), size_t((Wsize) * sizeof(devPtrW[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrY), size_t((Ysize) * sizeof(devPtrY[0]))));

        hostX = (T_ELEM*)calloc(size_t(Xsize), sizeof(hostX[0]));
        hostW = (T_ELEM*)calloc(size_t(Wsize), sizeof(hostW[0]));
        hostY = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostY[0]));
        host_ref = (T_ELEM*)calloc(size_t(ref_size), sizeof(host_ref[0]));

        initImage(hostX, Xsize);
        initImage(hostW, Wsize);
        initImage(hostY, Ysize);

        checkCudaErr(cudaMemcpy(devPtrX, hostX, size_t(sizeof(hostX[0]) * Xsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrW, hostW, size_t(sizeof(hostW[0]) * Wsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrY, hostY, size_t(sizeof(hostY[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());
    }

    explicit SurfaceManager(int64_t Xsize, int64_t Wsize, int64_t Ysize, int64_t Bsize, bool isConvBiasAdd) {
        (void)isConvBiasAdd;

        checkCudaErr(cudaMalloc((void**)&(devPtrX), size_t((Xsize) * sizeof(devPtrX[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrW), size_t((Wsize) * sizeof(devPtrW[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrY), size_t((Ysize) * sizeof(devPtrY[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrZ), size_t((Ysize) * sizeof(devPtrZ[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrB), size_t((Bsize) * sizeof(devPtrB[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrAfterConv), size_t((Ysize) * sizeof(devPtrAfterConv[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrAfterAdd), size_t((Ysize) * sizeof(devPtrAfterAdd[0]))));
        checkCudaErr(cudaMalloc((void**)&(devPtrAfterBias), size_t((Ysize) * sizeof(devPtrAfterBias[0]))));

        hostX = (T_ELEM*)calloc(size_t(Xsize), sizeof(hostX[0]));
        hostW = (T_ELEM*)calloc(size_t(Wsize), sizeof(hostW[0]));
        hostY = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostY[0]));
        hostZ = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostZ[0]));
        hostB = (T_ELEM*)calloc(size_t(Bsize), sizeof(hostB[0]));
        hostAfterConv = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostAfterConv[0]));
        hostAfterAdd = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostAfterAdd[0]));
        hostAfterBias = (T_ELEM*)calloc(size_t(Ysize), sizeof(hostAfterBias[0]));
        host_ref = (T_ELEM*)calloc(size_t(Ysize), sizeof(host_ref[0]));

        initImage(hostX, Xsize);
        initImage(hostW, Wsize);
        initImage(hostY, Ysize);
        initImage(hostZ, Ysize);
        initImage(hostB, Bsize);
        initImage(hostAfterAdd, Ysize);
        initImage(hostAfterBias, Ysize);
        initImage(hostAfterConv, Ysize);

        checkCudaErr(cudaMemcpy(devPtrX, hostX, (size_t)(sizeof(hostX[0]) * Xsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrW, hostW, (size_t)(sizeof(hostW[0]) * Wsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrY, hostY, (size_t)(sizeof(hostY[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrZ, hostZ, (size_t)(sizeof(hostZ[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrB, hostB, (size_t)(sizeof(hostB[0]) * Bsize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrAfterAdd, hostAfterAdd, (size_t)(sizeof(hostAfterAdd[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrAfterBias, hostAfterBias, (size_t)(sizeof(hostAfterBias[0]) * Ysize), cudaMemcpyHostToDevice));
        checkCudaErr(cudaMemcpy(devPtrAfterConv, hostAfterConv, (size_t)(sizeof(hostAfterConv[0]) * Ysize), cudaMemcpyHostToDevice));

        checkCudaErr(cudaDeviceSynchronize());
    }

    ~SurfaceManager() {
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
        if (host_ref) free(host_ref);
    }

};


template <typename T_ELEM>
struct Surface {
    T_ELEM* devPtr = NULL;
    T_ELEM* hostPtr = NULL;
    T_ELEM* hostRefPtr = NULL;
    int64_t n_elems = 0;

    explicit Surface(int64_t n_elems, bool hasRef) : n_elems(n_elems) {
        checkCudaErr(cudaMalloc((void**)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));
        hostPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostPtr[0]));
        if (hasRef) {
            hostRefPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostRefPtr[0]));
        }
        initImage(hostPtr, n_elems);
        checkCudaErr(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());
    }

    explicit Surface(int64_t n_elems, bool hasRef, bool isInterleaved) {
        (void)isInterleaved;
        checkCudaErr(cudaMalloc((void**)&(devPtr), (n_elems) * sizeof(devPtr[0])));
        hostPtr = (T_ELEM*)calloc(n_elems, sizeof(hostPtr[0]));
        if (hasRef) {
            hostRefPtr = (T_ELEM*)calloc(n_elems, sizeof(hostRefPtr[0]));
        }
        initImage(hostPtr, n_elems);
        uint32_t* temp = (uint32_t*)hostPtr;
        for (size_t i = 0; i < n_elems; i = i + 2) {
            temp[i + 1] = 1u;
        }

        checkCudaErr(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());
    }

    explicit Surface(int64_t size, bool hasRef, T_ELEM fillValue) : n_elems(size) {
        checkCudaErr(cudaMalloc((void**)&(devPtr), (size) * sizeof(devPtr[0])));
        hostPtr = (T_ELEM*)calloc(size, sizeof(hostPtr[0]));
        if (hasRef) {
            hostRefPtr = (T_ELEM*)calloc(n_elems, sizeof(hostRefPtr[0]));
        }
        for (int i = 0; i < size; i++) {
            hostPtr[i] = fillValue;
        }
        checkCudaErr(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
        checkCudaErr(cudaDeviceSynchronize());
    }

    ~Surface() {
        if (devPtr) {
            cudaFree(devPtr);
            devPtr = nullptr;
        }
        if (hostPtr) {
            free(hostPtr);
            hostPtr = nullptr;
        }
        if (hostRefPtr) {
            free(hostRefPtr);
            hostRefPtr = nullptr;
        }
    }
};


// Convert a linear index
// i = d_1 s_1 ... s_n + d_2 s_2 ... s_n + d_n-1 s_n + d_n
// into a multidimensional index
// (d_1, d_2, ..., d_n)
void lin2dim(int64_t id, int64_t* ids, const int64_t* dims, int64_t length) {
    int64_t idrem = id;
    int64_t prod = 1;  // accumulates the product of the dimensions
    for (int64_t i = length - 1; i >= 0; i--) {
        ids[i] = (idrem / prod) % dims[i];
        idrem = id - ids[i] * prod;
        prod *= dims[i];
    }
}

// Convert a multidimensional index
// (d_1, d_2, ..., d_n)
// into a linear index
// i = d_1 s_1 + ... + d_n s_n
int64_t dim2lin(const int64_t* ids, const int64_t* strides, int64_t length) {
    int64_t res = 0;
    for (int64_t i = 0; i < length; i++) {
        res += ids[i] * strides[i];
    }
    return static_cast<int>(res);
}
void doEpilog(float* out, int64_t idx, float alphaAcc, float beta) {
    if (beta == 0.f) {
        out[idx] = alphaAcc;
    }
    else {
        out[idx] = alphaAcc + out[idx] * beta;
    }
}

//void doEpilog(half1* out, int64_t idx, float alphaAcc, float beta) {
//    if (beta == 0.f) {
//        out[idx] = cpu_float2half_rn(alphaAcc);
//    }
//    else {
//        out[idx] = cpu_float2half_rn(alphaAcc + cpu_half2float(out[idx]) * beta);
//    }
//}

void doEpilog(int8_t* out, int64_t idx, int32_t alphaAcc, float beta) {
    int32_t val;
    if (beta == 0.f) {
        val = alphaAcc;
    }
    else {
        val = alphaAcc + int(float(out[idx]) * beta);
    }
    // Properly handle overflow errors in the same way cuDNN does
    if (val > 127) {
        val = 127;
    }
    else if (val < -128) {
        val = -128;
    }
    out[idx] = static_cast<int8_t>(val);
}


float getError(float dev, float ref) {
    if (ref > 1.0 || ref < -1.0)
        return (dev - ref) / ref;
    else
        return dev - ref;
}

//float getError(half1 dev, half1 ref) {
//    if (cpu_half2float(ref) > 1.0 || cpu_half2float(ref) < -1.0)
//        return (cpu_half2float(dev) - cpu_half2float(ref)) / cpu_half2float(ref);
//    else
//        return cpu_half2float(dev) - cpu_half2float(ref);
//}

int8_t getError(int8_t dev, int8_t ref) {
    return dev - ref;
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


// T_ELEM is the type the data is stored in, T_MATH is the type the calculations are done in.
template <typename T_ELEM, typename T_MATH>
void conv_cpu_ref(
    const T_ELEM* inputData,
    const T_ELEM* filterData,
    T_ELEM* outputData,
    int resizeFactor,
    cudnnTensorFormat_t filterFormat,
    const int64_t* inDims,
    const int64_t* filDims,
    const int64_t* diffDims,
    const int64_t* stride,
    const int64_t* pad,
    const int64_t* dilation,
    int64_t nbDims)
{
    int64_t imDims = nbDims - 2;
    float alpha = 1.0f;
    float beta = 0.0;
    // Some sanity checks
    // image   is n x c x h x w
    // diff    is n x k x p x q
    // filter  is k x c x r x s
    assert(inDims[0] == diffDims[0]);
    assert(inDims[1] == filDims[1]);
    assert(diffDims[1] == filDims[0]);

    // Filter stride
    int64_t filterStride[8];
    int64_t inStride[8];
    int64_t diffStride[8];

    generateStrides(inDims, inStride, nbDims, filterFormat);
    generateStrides(diffDims, diffStride, nbDims, filterFormat);
    generateStrides(filDims, filterStride, nbDims, filterFormat);

    int64_t filStride[8] = { 0 };
    generateStrides(filDims, filStride, nbDims, filterFormat);

    bool isConv = true;  //(CUDNN_CONVOLUTION == mode) ;

    // Number of pixels in output
    int64_t nPixelsOut = 1;
    for (int i = 2; i < nbDims; i++) {
        nPixelsOut *= diffDims[i];
    }

    // Number of pixels in filter
    int64_t nPixelsFil = 1;
    for (int i = 2; i < nbDims; i++) {
        nPixelsFil *= filDims[i];
    }

    // Used to store coordinates
    int64_t filIds[8] = { 0 };
    int64_t outIds[8] = { 0 };
    int64_t inIds[8] = { 0 };
    int64_t tmpIds[8] = { 0 };

    // For each image in the output
    for (int64_t ni = 0; ni < diffDims[0]; ni++) {
        // For each outer feature layer of the output image
        for (int ki_outer = 0; ki_outer < diffDims[1] / resizeFactor; ki_outer++) {
            int64_t outputOffset = ni * diffStride[0] / resizeFactor + ki_outer * diffStride[1];
            // For every pixel in this output image's feature layer
            for (int outId = 0; outId < nPixelsOut; outId++) {
                // Get output pixel ids
                lin2dim(outId, outIds, diffDims + 2, imDims);  // Skip n and k dimensions
                // Now we get the coordinates in input space of the "top left" corner 
                // of the filter: multiply by stride and remove pad
                for (int d = 0; d < imDims; d++) {
                    inIds[d] = outIds[d] * stride[d] - pad[d];
                }
                // For each inner feature layer of the output image
                for (int ki_inner = 0; ki_inner < resizeFactor; ki_inner++) {
                    // We prepare to accumulate
                    T_MATH tmp = 0;
                    // For each outer feature layer of the input image and filter
                    for (int ci = 0; ci < inDims[1] / resizeFactor; ci++) {
                        int64_t inputOffset = ni * inStride[0] / resizeFactor + ci * inStride[1];
                        int64_t filterOffset = (ki_outer * resizeFactor + ki_inner) * filStride[0] / resizeFactor + ci * filStride[1];
                        // Now for every pixel in the filter
                        for (int filId = 0; filId < nPixelsFil; filId++) {
                            // Get the position of the pixel
                            lin2dim(filId, filIds, filDims + 2, imDims);
                            // Compute the corresponding output pixel
                            // and check whether we are in the padding area on the fly too
                            // (not that for convolution, we flip the image patch;
                            // equivalent to flipping the filter patch).
                            bool inside = true;
                            for (int d = 0; d < imDims && inside; d++) {
                                if (isConv) {
                                    tmpIds[d] = inIds[d] + dilation[d] * (filDims[2 + d] - 1 - filIds[d]);
                                }
                                else {
                                    tmpIds[d] = inIds[d] + dilation[d] * filIds[d];
                                }
                                // If we are in the padding area: stop and skip computations
                                inside &= (tmpIds[d] >= 0 && tmpIds[d] < inDims[2 + d]);
                            }
                            if (inside) {
                                int64_t actualTmpId = inputOffset + dim2lin(tmpIds, (inStride)+2, imDims);
                                // int actualFilId = filterOffset + filId ;
                                int64_t actualFilId = filterOffset + dim2lin(filIds, (filStride)+2, imDims);

                                // For each inner feature layer of the input image and filter
                                for (int i = 0; i < resizeFactor; i++) {
                                    T_ELEM fval = filterData[(size_t)(actualFilId * resizeFactor + i)];
                                    T_ELEM ival = inputData[(size_t)(actualTmpId * resizeFactor + i)];
                                    tmp = doFma(fval, ival, tmp);
                                }
                            }
                        }
                    }

                    // Store final result in proper position in output image
                    int64_t actualOutId = outputOffset + dim2lin(outIds, (diffStride)+2, imDims);
                    doEpilog(outputData, actualOutId * resizeFactor + ki_inner, alpha * tmp, beta);
                }
            }
        }
    }
}

void Test()
{
    std::cout << "TEST_CASE :: Use heuristics for engine generation" << std::endl;
    //INFO("TEST_CASE :: Use heuristics for engine generation");
    int64_t dimA[] = { 8, 32, 4, 4 };
    int64_t filterdimA[] = { 32, 32, 1, 1 };
    int64_t outdimA[] = { 0, 0, 0, 0 }; // Computed Below
    int64_t padA[] = { 0, 0 };
    int64_t dilationA[] = { 1, 1 };
    int64_t convstrideA[] = { 1, 1 };

    int numErrors = 0;

    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for (int dim = 0; dim < 2; dim++) {
        outdimA[dim + 2] = getFwdConvOutputDim(dimA[dim + 2], padA[dim], filterdimA[dim + 2], convstrideA[dim], dilationA[dim]);
    }


    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;

    printf("====DIMENSIONS====\n");
    printf("input dims are %lld, %lld, %lld, %lld\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %lld, %lld, %lld, %lld\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %lld, %lld, %lld, %lld\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);


    int64_t Xsize = dimA[0] * dimA[1] * dimA[2] * dimA[3];
    int64_t Wsize = filterdimA[0] * filterdimA[1] * filterdimA[2] * filterdimA[3];
    int64_t Ysize = outdimA[0] * outdimA[1] * outdimA[2] * outdimA[3];

    SurfaceManager<float> sm(Xsize, Wsize, Ysize, Ysize);

    run_from_heuristics(dimA, padA, convstrideA, dilationA, filterdimA, outdimA, CUDNN_DATA_FLOAT, mode, sm.devPtrX, sm.devPtrW, sm.devPtrY, CUDNN_HEUR_MODE_INSTANT);

    cudaDeviceSynchronize();
    cudaMemcpy(sm.hostY, sm.devPtrY, (size_t)(sizeof(sm.hostY[0]) * Ysize), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    conv_cpu_ref<float, float>(sm.hostX, sm.hostW, sm.host_ref, 1, CUDNN_TENSOR_NCHW, dimA, filterdimA, outdimA, convstrideA, padA, dilationA, 4/*Dims*/);

    for (int index = 0; index < Ysize; index++) {  // assuming in data is packed
        float diff = getError(sm.hostY[index], sm.host_ref[index]);
        if (diff < 0) diff = -diff;
        if (diff > THRESHOLD) { numErrors++; }
    }
    assert(numErrors == 0);

    std::cout << "\n========================================================================================\n";
}