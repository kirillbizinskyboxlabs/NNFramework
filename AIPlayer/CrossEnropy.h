#pragma once
#include "Layer.h"

import <ranges>;

class CrossEntropy : public Layer
{
public:
	CrossEntropy(cudnnHandle_t& handle, 
		Layer* previousLayer, 
		const Hyperparameters& hyperparameters, 
		bool verbose = false, 
		std::string name = "CrossEntropy");
	~CrossEntropy();

	void printOutput() override;
	void printLoss();
	void printGrad() override;

	//int executeInference() override { return 0; } // stubbed out since inference make little sense. Loss shouldn't be a Layer probably...
	//void propagateForward() override;
	void propagateBackward() override;

	void calculateLoss();
	void calculateGrad();

	void setLabel(std::span<uint8_t> labels);
	float* getLabelDataPtr();
	void syncLabel();

private:
	void _initLoss();
	void _initGrad();
	cudnn_frontend::Tensor _flattenTensor(cudnn_frontend::Tensor& tensor);

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