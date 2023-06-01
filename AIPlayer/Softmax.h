#pragma once
#include "Layer.h"


class Softmax : public Layer
{
public:
	Softmax(cudnnHandle_t& handle, 
		Layer* previousLayer,
		const Hyperparameters& hyperparameters,
		bool verbose = false, 
		std::string name = "Softmax");

	//int executeInference() override;

	void printOutput() override;

	void propagateForward() override;
	void propagateBackward() override;

private:
	cudnnTensorDescriptor_t srcTensorDesc;
	cudnnTensorDescriptor_t sftTensorDesc;

	std::vector<int> mDims;
};