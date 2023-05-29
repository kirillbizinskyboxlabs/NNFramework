#include "NeuralNetwork.h"

#include "ConvBiasAct.h"
#include "CrossEnropy.h"
#include "Input.h"
#include "Pool.h"
#include "Softmax.h"

NeuralNetwork::NeuralNetwork(size_t batchSize, 
							 size_t nbDims, 
							 size_t inputDims[], 
							 VERBOSITY verbosity)
	: mBatchSize(batchSize)
	, mVerbosity(verbosity)
	, mNbDims(nbDims + 1) // + batch 
	, mInputDims(mNbDims)
{
	mInputDims[0] = mBatchSize;

	Utils::checkCudnnError(cudnnCreate(&mHandle));

	for (size_t d = 0; d < nbDims; ++d)
	{
		mInputDims[d + 1] = inputDims[d];
	}

	mLayers.emplace_back(new Input(mHandle, mNbDims, mInputDims.data()));
}

NeuralNetwork::~NeuralNetwork()
{
	for (auto&& layer : mLayers)
	{
		delete layer;
		layer = nullptr; // ??? do we need it ???
	}

	if (mHandle) 
	{
		cudnnDestroy(mHandle);
	}
}

void NeuralNetwork::addConvBiasAct(const int64_t kernelSize, const int64_t filterSize, const int64_t convPad, bool verbose, std::string name)
{
	bool needDataGrad = mLayers.size() > 1;
	bool mTraining = true; // TODO: move to member var
	mLayers.emplace_back(new ConvBiasAct(mHandle, kernelSize, filterSize, mLayers.back(), mLearningRate, convPad, mTraining, needDataGrad, verbose, std::move(name)));
}

void NeuralNetwork::addPool(bool verbose, std::string name)
{
	mLayers.emplace_back(new Pool(mHandle, mLayers.back(), verbose, std::move(name)));
}

void NeuralNetwork::addSoftmax(bool verbose, std::string name)
{
	mLayers.emplace_back(new Softmax(mHandle, mLayers.back(), verbose, std::move(name)));
}

void NeuralNetwork::addCrossEntropy(bool verbose, std::string name)
{
	mLayers.emplace_back(new CrossEntropy(mHandle, mLayers.back(), verbose, std::move(name)));
}

float* NeuralNetwork::getInputDataPtr()
{
	return static_cast<Input*>(mLayers[0])->getHostPtr();
}

float* NeuralNetwork::getLabelDataPtr()
{
	return static_cast<CrossEntropy*>(mLayers.back())->getLabelDataPtr();
}

void NeuralNetwork::syncLabel()
{
	return static_cast<CrossEntropy*>(mLayers.back())->syncLabel();
}

void NeuralNetwork::setLabel(std::span<uint8_t> labels)
{
	static_cast<CrossEntropy*>(mLayers.back())->setLabel(labels);
}

void NeuralNetwork::train()
{
	for (auto&& layer : mLayers)
	{
		layer->propagateForward();
		
		if (mVerbosity == VERBOSITY::MAX)
		{
			layer->printOutput();
		}
	}

	if (mVerbosity == VERBOSITY::LEVEL1)
	{
		//static_cast<CrossEntropy*>(mLayers.back())->printLoss();
		printLoss();
	}

	mLearningRate = static_cast<float>(0.0001 * pow((1.0 + 0.0001 * mIter++), (-0.75)));

	for (auto it = mLayers.rbegin(); it != mLayers.rend(); ++it)
	{
		(*it)->propagateBackward();
		if (mVerbosity == VERBOSITY::MAX)
		{
			(*it)->printGrad();
		}
	}
}

void NeuralNetwork::printLoss()
{
	static_cast<CrossEntropy*>(mLayers.back())->printLoss();
}
