#pragma once

#include "DevUtils.h"

import <vector>;
import <memory>;
import <string>;
import <filesystem>;

class Layer
{
public:
	Layer(cudnnHandle_t& handle, 
        Layer* previousLayer,
        const Hyperparameters& hyperparameters,
        std::string name = "",
        VERBOSITY verbosityLevel = VERBOSITY::ERROR);
	virtual ~Layer();

    template <typename T_ELEM>
    struct Surface 
    {
        //using namespace Utils;

        T_ELEM* devPtr = nullptr;
        T_ELEM* hostPtr = nullptr;
        int64_t n_elems = 0;

        explicit Surface(int64_t n_elems, bool needHostPtr = true) 
            : n_elems(n_elems) 
        {
            Utils::checkCudaError(cudaMalloc((void**)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));

            if (needHostPtr)
            {
                hostPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostPtr[0]));
                Utils::initImage(hostPtr, n_elems); // default random initialization
                Utils::checkCudaError(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
                Utils::checkCudaError(cudaDeviceSynchronize());
            }
        }

        explicit Surface(int64_t n_elems, T_ELEM fillValue)
            : n_elems(n_elems)
        {
            Utils::checkCudaError(cudaMalloc((void**)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));
            hostPtr = (T_ELEM*)calloc(n_elems, sizeof(hostPtr[0]));
            for (int i = 0; i < n_elems; i++) {
                hostPtr[i] = fillValue;
            }
            Utils::checkCudaError(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
            Utils::checkCudaError(cudaDeviceSynchronize());
        }

        explicit Surface(int64_t n_elems, std::function<void(size_t, T_ELEM*)> initializer)
            : n_elems(n_elems)
        {
            Utils::checkCudaError(cudaMalloc((void**)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));
            hostPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostPtr[0]));
            initializer((size_t)n_elems, hostPtr);
            Utils::checkCudaError(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
            Utils::checkCudaError(cudaDeviceSynchronize());
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
        }

        void devToHostSync() {
            if (!hostPtr) return;
            cudaDeviceSynchronize();
            Utils::checkCudaError(cudaMemcpy(hostPtr, devPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
        }

        void hostToDevSync() {
            if (!hostPtr) return;
            cudaDeviceSynchronize();
            Utils::checkCudaError(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();
        }
    };

    const cudnn_frontend::Tensor& getOutputTensor() const;
    Surface<float>& getOutputSurface() const; // does this function makes sense to be const? Surface is expected to change
    Surface<float>& getGradSurface() const;

    //virtual int executeInference();
    virtual void propagateForward();
    virtual void propagateBackward() = 0;

    virtual void printOutput();
    virtual void printGrad();

    virtual void saveParameters(const std::filesystem::path& dir, std::string_view NeuralNetworkName) {} // do nothing by default
    virtual void loadParameters(const std::filesystem::path& dir, std::string_view NeuralNetworkName) {} // do nothing by default

protected:
    static int64_t generateTensorId()
    {
        static int64_t id = 0;
        return id++;
    }

	void _setPlan(std::vector<cudnn_frontend::Operation const*>& ops,
		          std::vector<void*>& data_ptrs,
		          std::vector<int64_t>& uids,
		          std::unique_ptr<cudnn_frontend::ExecutionPlan>& plan,
		          std::unique_ptr<cudnn_frontend::VariantPack>& variant,
		          int64_t& workspace_size,
		          void*& workspace_ptr);

    void _setForwardPropagationPlan(std::vector<cudnn_frontend::Operation const*>& ops,
                                    std::vector<void*>& data_ptrs,
                                    std::vector<int64_t>& uids);

	cudnnHandle_t& mHandle;
    VERBOSITY mVerbosityLevel;

	std::unique_ptr<Surface<float>> mOutputSurface;
	std::unique_ptr<Surface<float>> mGradSurface;

	std::unique_ptr<cudnn_frontend::Tensor> mOutputTensor; // outputTensor we will share it. does it justify using shared pointer? TODO: rethink whether to follow general notation or using a more meaningful name
	std::unique_ptr<cudnn_frontend::Tensor> mGradTensor; // it's the same as output tensor

	std::unique_ptr<cudnn_frontend::ExecutionPlan> mForwardPropagationPlan;
	std::unique_ptr<cudnn_frontend::VariantPack> mForwardPropagationVariantPack;

	std::unique_ptr<cudnn_frontend::ExecutionPlan> mDataGradPlan;
	std::unique_ptr<cudnn_frontend::VariantPack> mDataGradVariantPack;

    int64_t mDataGradWorkspaceSize;
    void* mDataGradWorkspacePtr;

	int64_t mForwardPropagationWorkspaceSize;
	void* mForwardPropagationWorkspacePtr;

    Layer* mPreviousLayer;

	std::string mName;

    const Hyperparameters& mHyperparameters;
};