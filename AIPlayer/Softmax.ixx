export module NeuralNetwork:Softmax;

import :Layer;

export class Softmax : public Layer
{
public:
	Softmax(cudnnHandle_t& handle,
		Layer* previousLayer,
		const Hyperparameters& hyperparameters,
		std::string name = "Softmax",
		VERBOSITY verbosity = VERBOSITY::MIN);

	//int executeInference() override;

	void printOutput() override;

	void propagateForward() override;
	void propagateBackward() override;

private:
	cudnnTensorDescriptor_t srcTensorDesc;
	cudnnTensorDescriptor_t sftTensorDesc;

	std::vector<int> mDims;
};