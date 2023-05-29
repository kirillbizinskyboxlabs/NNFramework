#pragma once

//#include <cudnn.h>
#include "DevUtils.h"

import <vector>;
import <memory>;
import <string>;

//class cudnnHandle_t;

//namespace cudnn_frontend
//{
//	class Tensor;
//	class Operation;
//	class ExecutionPlan;
//	class VariantPack;
//}

class Layer
{
public:
	Layer(cudnnHandle_t& handle, Layer* previousLayer, bool verbose = false, std::string name = "");
	virtual ~Layer();

    template <typename T_ELEM>
    struct Surface 
    {
        //using namespace Utils;

        T_ELEM* devPtr = nullptr;
        T_ELEM* hostPtr = nullptr;
        int64_t n_elems = 0;

        explicit Surface(int64_t n_elems) 
            : n_elems(n_elems) 
        {
            Utils::checkCudaError(cudaMalloc((void**)&(devPtr), (size_t)((n_elems) * sizeof(devPtr[0]))));
            hostPtr = (T_ELEM*)calloc((size_t)n_elems, sizeof(hostPtr[0]));
            Utils::initImage(hostPtr, n_elems);
            Utils::checkCudaError(cudaMemcpy(devPtr, hostPtr, size_t(sizeof(hostPtr[0]) * n_elems), cudaMemcpyHostToDevice));
            Utils::checkCudaError(cudaDeviceSynchronize());
        }

        explicit Surface(int64_t size, T_ELEM fillValue) 
            : n_elems(size) 
        {
            Utils::checkCudaError(cudaMalloc((void**)&(devPtr), (size) * sizeof(devPtr[0])));
            hostPtr = (T_ELEM*)calloc(size, sizeof(hostPtr[0]));
            for (int i = 0; i < size; i++) {
                hostPtr[i] = fillValue;
            }
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
            cudaDeviceSynchronize();
            Utils::checkCudaError(cudaMemcpy(hostPtr, devPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
        }

        void hostToDevSync() {
            cudaDeviceSynchronize();
            Utils::checkCudaError(cudaMemcpy(devPtr, hostPtr, sizeof(hostPtr[0]) * n_elems, cudaMemcpyHostToDevice));
            cudaDeviceSynchronize();
        }
    };

    //virtual int executeInference();
    virtual void propagateForward();
    virtual void propagateBackward() = 0;

    cudnn_frontend::Tensor& getOutputTensor() const; // Is this OK when const return non const reference that is known to be changed elsewhere?
    Surface<float>& getOutputSurface() const; //TODO: same question
    Surface<float>& getGradSurface() const; // maybe make it a struct even?

    virtual void printOutput();
    virtual void printGrad();

    // temporary public??
    static int64_t generateTensorId()
    {
        static int64_t id = 0;
        return id++;
    }

protected:
	void _setPlan(std::vector<cudnn_frontend::Operation const*>& ops,
		std::vector<void*> data_ptrs,
		std::vector<int64_t> uids,
		std::unique_ptr<cudnn_frontend::ExecutionPlan>& plan,
		std::unique_ptr<cudnn_frontend::VariantPack>& variant,
		int64_t& workspace_size,
		void*& workspace_ptr);

    void _setForwardPropagationPlan(std::vector<cudnn_frontend::Operation const*>& ops,
        std::vector<void*> data_ptrs,
        std::vector<int64_t> uids);

	cudnnHandle_t& mHandle;
	bool mVerbose;

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
};