#pragma once

import <vector>;
import <string>;
import <ranges>;

#include <cudnn.h>
#include "DevUtils.h"

class Layer;

class NeuralNetwork
{
public:
	NeuralNetwork(size_t batchSize, 
				  size_t nbDims, 
				  size_t inputDims[], 
				  VERBOSITY verbosity = VERBOSITY::MIN,
				  std::string name = "");
	~NeuralNetwork();

	void addConvBiasAct(const int64_t kernelSize,
						const int64_t filterSize,
						const int64_t convPad = 2,
						bool verbose = false,
						std::string name = "ConvBiasAct");

	void addPool(bool verbose = false, std::string name = "Pool");

	void addSoftmax(bool verbose = false, std::string name = "Softmax");
	void addCrossEntropy(bool verbose = false, std::string name = "CrossEntropy");

	float* getInputDataPtr();
	float* getLabelDataPtr();
	void syncData();
	void syncLabel();
	void setLabel(std::span<uint8_t> labels); // should it also be a responsibility of a user??

	void train();
	void inference();

	void printLoss();
	void printOutput();

	float getLoss();

	// temporary public
	Hyperparameters mHyperparameters;

	void saveParameters();
	void loadParameters();

private:
	cudnnHandle_t mHandle;

	std::vector<Layer*> mLayers;
	int64_t mBatchSize;
	int64_t mNbDims;
	std::vector<int64_t> mInputDims; //?

	VERBOSITY mVerbosity;

	float mLearningRate;
	size_t mIter = 0;

	std::string mName;
};

