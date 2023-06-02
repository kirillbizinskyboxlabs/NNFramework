#pragma once
#include "Layer.h"

class ConvBiasAct : public Layer
{
public:
	ConvBiasAct(cudnnHandle_t& handle,
		const int64_t kernelSize,
		const int64_t filterSize,
		Layer* previousLayer,
		const Hyperparameters& hyperparameters,
		const float& learningRate,
		const int64_t convPad = 2,
		bool training = true,
		bool needDataGrad = true,
		bool verbose = false,
		std::string name = "ConvBiasAct",
		VERBOSITY verbosity = VERBOSITY::MIN);
	ConvBiasAct(const ConvBiasAct&) = delete;
	ConvBiasAct& operator=(const ConvBiasAct&) = delete;
	~ConvBiasAct() = default;

	void propagateBackward() override;
private:
	void _setupBackPropagation(bool needDataGrad);
	void _setupBiasBackPropagation();
	void _setupWeightBackPropagation();
	void _setupDataBackPropagation();

	void _printBias();
	void _printFilter();
	void _printActivationGrad();
	void _printBiasGrad();
	void _printFilterGrad();

	std::unique_ptr<Surface<float>> mWeightsSurface;
	std::unique_ptr<Surface<float>> mBiasSurface;

	//cudnnReduceTensorDescriptor_t mReduceTensorDesc;
	cudnnTensorDescriptor_t mInputTensorDesc; // TODO: acquire from previous Layer
	cudnnTensorDescriptor_t mGradTensorDesc; // TODO: move to Layer
	cudnnTensorDescriptor_t mBiasGradTensorDesc;
	cudnnFilterDescriptor_t mFilterDesc;
	cudnnConvolutionDescriptor_t mConvDesc;
	cudnnConvolutionBwdDataAlgoPerf_t mBwdDPerf;
	cudnnConvolutionBwdFilterAlgoPerf_t mBwdFPerf;

	size_t mBiasGradWorkspaceSize;
	void* mBiasGradWorkspacePtr;

	//size_t mGradWorkspaceSize;
	//void* mGradWorkspacePtr;

	//float mLearningRate = 0.001;
	const float& mLearningRate;
	bool mNeedDataGrad = true;

	std::unique_ptr<Surface<float>> mBiasGradSurface;
	std::unique_ptr<Surface<float>> mWeightsGradSurface;
	std::unique_ptr<Surface<float>>	mActivationGradSurface;

	std::unique_ptr<cudnn_frontend::ExecutionPlan> mActivationGradPlan;
	std::unique_ptr<cudnn_frontend::VariantPack> mActivationGradVariantPack;

	int64_t mActivationGradWorkspaceSize;
	void* mActivationGradWorkspacePtr;

	struct // mSGD parameters
	{
		//float* d_v_f = nullptr;
		//float* d_v_b = nullptr;
		std::unique_ptr<Surface<float>> mGradBiasVelocitySurface;
		std::unique_ptr<Surface<float>> mGradFilterVelocitySurface;
	} mSGD;
};

