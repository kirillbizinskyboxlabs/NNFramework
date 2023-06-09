export module NeuralNetwork:CrossEntropy;

import :Layer;

import <ranges>;

export class CrossEntropy : public Layer
{
public:
	CrossEntropy(cudnnHandle_t& handle,
		Layer* previousLayer,
		const Hyperparameters& hyperparameters,
		std::string name = "CrossEntropy",
		VERBOSITY verbosity = VERBOSITY::MIN);
	~CrossEntropy();

	void printOutput() override;
	void printLoss();
	void printGrad() override;

	float getLoss();

	void propagateBackward() override;

	void calculateLoss();
	void calculateGrad();

	void setLabel(std::span<uint8_t> labels);
	float* getLabelDataPtr();
	void syncLabel();

private:
	void _initLoss();
	void _initGrad();

	int64_t mBatchSize;
	int64_t mNumClasses;

	std::unique_ptr<Surface<float>> mProductSurface; // matmul output holder, internal

	std::unique_ptr<Surface<float>> mLabelSurface;
	std::unique_ptr<Surface<float>> mLossSurface;

	std::unique_ptr<cudnn_frontend::ExecutionPlan> mGradPlan;
	std::unique_ptr<cudnn_frontend::VariantPack> mGradVariantPack;

	int64_t mGradWorkspaceSize;
	void* mGradWorkspacePtr;
};