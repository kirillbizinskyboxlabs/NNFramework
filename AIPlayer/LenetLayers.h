#pragma once
//#include <cudnn_frontend.h>
#include "LeNetHelpers.h"
#include <vector>
#include <memory>
#include <ranges>
#include <string>

class Layer
{
public:
	Layer(cudnnHandle_t& handle, bool verbose = false, std::string name = "");
	virtual ~Layer();

	virtual int executeInference();

	cudnn_frontend::Tensor& getOutputTensor() const; // Is this OK when const return non const reference that is known to be changed elsewhere?
	Helpers::Surface<float>& getOutputSurface() const; //TODO: same question
	Helpers::Surface<float>& getGradSurface() const; // maybe make it a struct even?

	virtual void printOutput();
	virtual void printGrad();

	// temporary public??
	static int64_t getTensorId()
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

	cudnnHandle_t& mHandle;
	bool mVerbose;

	std::unique_ptr<Helpers::Surface<float>> Y;
	std::unique_ptr<Helpers::Surface<float>> mGradSurface;

	std::unique_ptr<cudnn_frontend::Tensor> yTensor; // outputTensor we will share it. does it justify using shared pointer? TODO: rethink whether to follow general notation or using a more meaningful name
	std::unique_ptr<cudnn_frontend::Tensor> mGradTensor;

	//std::vector<cudnn_frontend::Operation const*> ops; // do we need it? doesn't look like we do
	std::unique_ptr<cudnn_frontend::ExecutionPlan> plan;
	std::unique_ptr<cudnn_frontend::VariantPack> variantPack;

	std::unique_ptr<cudnn_frontend::ExecutionPlan> mDataGradPlan;
	std::unique_ptr<cudnn_frontend::VariantPack> mDataGradVariantPack;

	int64_t workspace_size = 0;
	void* workspace_ptr = nullptr;

	std::string mName;
};

class ConvBiasAct : public Layer
{
public:
	ConvBiasAct(cudnnHandle_t& handle, 
				const int64_t* inputDim, 
				const int64_t kernelSize, 
				const int64_t filterSize, 
				cudnn_frontend::Tensor& inputTensor, 
				void* inputDevPtr,
				const int64_t convPad = 2,
				bool verbose = false);
	ConvBiasAct(const ConvBiasAct&) = delete;
	ConvBiasAct& operator=(const ConvBiasAct&) = delete;
private:
	std::unique_ptr<Helpers::Surface<float>> W;
	std::unique_ptr<Helpers::Surface<float>> B;

};


class Pool : public Layer
{
public:
	Pool(cudnnHandle_t& handle,
		cudnn_frontend::Tensor& inputTensor,
		void* inputDevPtr,
		bool verbose = false);
};

class FC : public Layer
{
public:
	FC(cudnnHandle_t& handle,
		Layer& prevLayer,
		int64_t numOutput,
		bool verbose = false,
		std::string name = "");

	void backpropagate();

	void printBias();
	void printWeights();

private:
	static cudnn_frontend::Tensor flattenTensor(cudnn_frontend::Tensor& tensor);
	
	void _setupGradient(int64_t biasDim[], int64_t weightsTensorDim[], const int64_t inputDim[]);
	void _setupBiasGrad(int64_t biasDim[]);
	void _setupWeightsGrad(int64_t weightsTensorDim[], const int64_t inputDim[]);
	void _setupDataGrad();

	std::unique_ptr<Helpers::Surface<float>> W; // weights
	std::unique_ptr<Helpers::Surface<float>> B; // bias
	std::unique_ptr<Helpers::Surface<float>> Z; // after bias

	//std::unique_ptr<cudnn_frontend::Tensor> mBiasTensor;
	std::unique_ptr<Helpers::Surface<float>> mBiasGradSurface;
	std::unique_ptr<cudnn_frontend::ExecutionPlan> mBiasGradPlan;
	std::unique_ptr<cudnn_frontend::VariantPack> mBiasGradVariantPack;
	int64_t mBiasGradWorkspaceSize = 0;
	void* mBiasGradWorkspacePtr = nullptr; // TODO: cleanup

	std::unique_ptr<Helpers::Surface<float>> mWeightsGradSurface;
	std::unique_ptr<cudnn_frontend::ExecutionPlan> mWeightsGradPlan;
	std::unique_ptr<cudnn_frontend::VariantPack> mWeightsGradVariantPack;
	int64_t mWeightsGradWorkspaceSize = 0;
	void* mWeightsGradWorkspacePtr = nullptr; // TODO: cleanup

	Layer& mPrevLayer; // better?
};

class Softmax : public Layer
{
public:
	//Softmax(cudnnHandle_t& handle,
	//	cudnn_frontend::Tensor& inputTensor,
	//	Helpers::Surface<float>& srcSurface,
	//	//void* inputDevPtr,
	//	bool verbose = false);

	Softmax(cudnnHandle_t& handle, Layer& prevLayer, bool verbose = false, std::string name = "Softmax");

	int executeInference() override;

	void printOutput() override;

	void backpropagate();

private:
	cudnnTensorDescriptor_t srcTensorDesc;
	cudnnTensorDescriptor_t sftTensorDesc;

	std::vector<int> mDims;

	//Helpers::Surface<float>& mSrcSurface; // TODO: rethink
	Layer& mPrevLayer; // better?
};

// Does it need to be a Layer?
class CrossEntropy : public Layer
{
public:
	//CrossEntropy(cudnnHandle_t& handle,
	//			 cudnn_frontend::Tensor& inputTensor,
	//			 void* inputDevPtr,
	//			 bool verbose = false);

	CrossEntropy(cudnnHandle_t& handle, Layer& prevLayer, bool verbose = false, std::string name = "CrossEntropy");
	~CrossEntropy();

	void printOutput() override;
	void printLoss();
	void printGrad() override;

	int executeInference() override { return 0; } // stubbed out since inference make little sense. Loss shouldn't be a Layer probably...

	void calculateLoss();
	void calculateGrad();

	//Helpers::Surface<float>& getGrad();

	//void setLabel(const std::vector<int8_t>& labels);
	void setLabel(std::span<uint8_t> labels);

private:

	//void _initLoss(cudnn_frontend::Tensor& inputTensor, void* inputDevPtr);
	//void _initGrad(cudnn_frontend::Tensor& inputTensor, void* inputDevPtr);
	void _initLoss();
	void _initGrad();


	int64_t mBatchSize;
	int64_t mNumClasses;

	std::unique_ptr<Helpers::Surface<float>> P; // matmul output holder, internal

	std::unique_ptr<Helpers::Surface<float>> L; // label
	std::unique_ptr<Helpers::Surface<float>> J; // loss
	//std::unique_ptr<Helpers::Surface<float>> G; // gradient // not needed anymore, acquired from prev layer

	//std::unique_ptr<cudnn_frontend::Tensor> mLabelTensor;
	//std::unique_ptr<cudnn_frontend::Tensor> mGradientTensor;
	std::unique_ptr<cudnn_frontend::ExecutionPlan> mGradPlan;
	std::unique_ptr<cudnn_frontend::VariantPack> mGradVariantPack;

	int64_t mGradWorkspaceSize = 0;
	void* mGradWorkspacePtr = nullptr;

	Layer& mPrevLayer; // better?
};