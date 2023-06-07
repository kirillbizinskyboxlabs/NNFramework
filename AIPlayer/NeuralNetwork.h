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
						std::string name = "ConvBiasAct",
						VERBOSITY verbosityOverride = VERBOSITY::NONE);

	void addPool(std::string name = "Pool", VERBOSITY verbosityOverride = VERBOSITY::NONE);
	void addSoftmax(std::string name = "Softmax", VERBOSITY verbosityOverride = VERBOSITY::NONE);
	void addCrossEntropy(std::string name = "CrossEntropy", VERBOSITY verbosityOverride = VERBOSITY::NONE);

	float* getInputDataPtr();
	float* getLabelDataPtr();
	void syncData();
	void syncLabel();

	//deprecated
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

