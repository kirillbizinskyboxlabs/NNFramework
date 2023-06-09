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
		std::string name = "ConvBiasAct",
		VERBOSITY verbosity = VERBOSITY::MIN,
		const int64_t dilation = 1,
		const int64_t convStride = 1);
	ConvBiasAct(const ConvBiasAct&) = delete;
	ConvBiasAct& operator=(const ConvBiasAct&) = delete;
	~ConvBiasAct();

	void propagateBackward() override;

	void saveParameters(const std::filesystem::path& dir, std::string_view NeuralNetworkName) override;
	void loadParameters(const std::filesystem::path& dir, std::string_view NeuralNetworkName) override;

private:
	void _setupBackPropagation();
	void _setupActivationBackPropagation();
	void _setupBiasBackPropagation();
	void _setupFilterBackPropagation();
	void _setupDataBackPropagation();
	void _setupBackPropagationAlgorithms();

	void _SGDUpdate();
	void _miniBatchSGDUpdate();

	void _printBias();
	void _printFilter();
	void _printActivationGrad();
	void _printBiasGrad();
	void _printFilterGrad();

	const int mConvDim = 2;
	const std::vector<int64_t> mPad;
	const std::vector<int64_t> mDilation;
	const std::vector<int64_t> mConvStride;
	const cudnnTensorFormat_t mTensorFormat = CUDNN_TENSOR_NHWC;
	const cudnnConvolutionMode_t mConvMode = CUDNN_CROSS_CORRELATION;
	const cudnnDataType_t mDataType = CUDNN_DATA_FLOAT;
	const float mAlpha = 1.0f;
	const float mBeta = 0.0f;
	const int64_t mFilterSize;
	const int64_t mKernelSize;

	std::unique_ptr<Surface<float>> mWeightsSurface;
	std::unique_ptr<Surface<float>> mBiasSurface;

	cudnnTensorDescriptor_t mInputTensorDesc; // TODO: acquire from previous Layer
	cudnnTensorDescriptor_t mGradTensorDesc; // TODO: move to Layer
	cudnnTensorDescriptor_t mBiasGradTensorDesc;
	cudnnFilterDescriptor_t mFilterDesc;
	cudnnConvolutionDescriptor_t mConvDesc;
	cudnnConvolutionBwdDataAlgoPerf_t mBwdDPerf;
	cudnnConvolutionBwdFilterAlgoPerf_t mBwdFPerf;

	size_t mBiasGradWorkspaceSize;
	void* mBiasGradWorkspacePtr;

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
		std::unique_ptr<Surface<float>> mGradBiasVelocitySurface;
		std::unique_ptr<Surface<float>> mGradFilterVelocitySurface;
	} mSGD;
};

