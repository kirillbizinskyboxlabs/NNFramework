module;

//#include "NeuralNetwork.h"

//#include "Layer.h"
#include "ConvBiasAct.h"
#include "CrossEnropy.h"
#include "Input.h"
#include "Pool.h"
#include "Softmax.h"

module NeuralNetwork;

//import :Layer;
//import :ConvBiasAct;
//import :CrossEnropy;
//import :Input;
//import :Pool;
//import :Softmax;

constexpr char SAVE_DIR_ROOT[] = "saved_weights";

NeuralNetwork::NeuralNetwork(size_t batchSize, 
							 size_t nbDims, 
							 size_t inputDims[], 
							 VERBOSITY verbosity,
							 std::string name)
	: mBatchSize(batchSize)
	, mVerbosity(verbosity)
	, mNbDims(nbDims + 1) // + batch 
	, mInputDims(mNbDims)
	, mName(std::move(name))
{
	mInputDims[0] = mBatchSize;

	Utils::checkCudnnError(cudnnCreate(&mHandle));

	for (size_t d = 0; d < nbDims; ++d)
	{
		mInputDims[d + 1] = inputDims[d];
	}

	mLayers.emplace_back(new Input(mHandle, mNbDims, mInputDims.data(), mHyperparameters, mVerbosity));
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

void NeuralNetwork::addConvBiasAct(const int64_t kernelSize, const int64_t filterSize, const int64_t convPad, std::string name, VERBOSITY verbosityOverride)
{
	verbosityOverride = verbosityOverride != VERBOSITY::NONE ? verbosityOverride : mVerbosity;
	bool needDataGrad = mLayers.size() > 1;
	bool mTraining = true; // TODO: move to member var
	mLayers.emplace_back(new ConvBiasAct(mHandle, kernelSize, filterSize, mLayers.back(), mHyperparameters, mLearningRate, convPad, mTraining, needDataGrad, std::move(name), verbosityOverride));
}

void NeuralNetwork::addPool(std::string name, VERBOSITY verbosityOverride)
{
	verbosityOverride = verbosityOverride != VERBOSITY::NONE ? verbosityOverride : mVerbosity;
	mLayers.emplace_back(new Pool(mHandle, mLayers.back(), mHyperparameters, std::move(name), verbosityOverride));
}

void NeuralNetwork::addSoftmax(std::string name, VERBOSITY verbosityOverride)
{
	verbosityOverride = verbosityOverride != VERBOSITY::NONE ? verbosityOverride : mVerbosity;
	mLayers.emplace_back(new Softmax(mHandle, mLayers.back(), mHyperparameters, std::move(name), verbosityOverride));
}

void NeuralNetwork::addCrossEntropy(std::string name, VERBOSITY verbosityOverride)
{
	verbosityOverride = verbosityOverride != VERBOSITY::NONE ? verbosityOverride : mVerbosity;
	mLayers.emplace_back(new CrossEntropy(mHandle, mLayers.back(), mHyperparameters, std::move(name), verbosityOverride));
}

float* NeuralNetwork::getInputDataPtr()
{
	return static_cast<Input*>(mLayers[0])->getHostPtr();
}

float* NeuralNetwork::getLabelDataPtr()
{
	return static_cast<CrossEntropy*>(mLayers.back())->getLabelDataPtr();
}

void NeuralNetwork::syncData()
{
	mLayers.front()->getOutputSurface().hostToDevSync();
}

void NeuralNetwork::syncLabel()
{
	return static_cast<CrossEntropy*>(mLayers.back())->syncLabel();
}

// deprecated
void NeuralNetwork::setLabel(std::span<uint8_t> labels)
{
	static_cast<CrossEntropy*>(mLayers.back())->setLabel(labels);
}

void NeuralNetwork::train()
{
	inference();

	if (mVerbosity >= VERBOSITY::MAX)
	{
		static_cast<CrossEntropy*>(mLayers.back())->printOutput();
		printLoss();
	}

	if (mHyperparameters.updateType == Hyperparameters::UpdateType::SGD)
	{
		// TODO: magic numbers -> hyperparameters
		mLearningRate = static_cast<float>(0.01 * pow((1.0 + 0.0001 * mIter++), (-0.75)));
	}

	for (auto it = mLayers.rbegin(); it != mLayers.rend(); ++it)
	{
		(*it)->propagateBackward();
		if (mVerbosity >= VERBOSITY::MAX)
		{
			(*it)->printGrad();
		}
	}
}

void NeuralNetwork::inference()
{
	for (auto&& layer : mLayers)
	{
		layer->propagateForward();

		if (mVerbosity >= VERBOSITY::MAX)
		{
			layer->printOutput();
		}
	}
}

void NeuralNetwork::printLoss()
{
	static_cast<CrossEntropy*>(mLayers.back())->printLoss();
}

void NeuralNetwork::printOutput()
{
	static_cast<CrossEntropy*>(mLayers.back())->printOutput(); // don't need to cast really
}

float NeuralNetwork::getLoss()
{
	return static_cast<CrossEntropy*>(mLayers.back())->getLoss();
}

void NeuralNetwork::saveParameters()
{
	std::filesystem::path root_dir = std::filesystem::current_path() / SAVE_DIR_ROOT / mName.c_str();

	try
	{
		std::filesystem::create_directories(root_dir);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	

	if (mVerbosity >= VERBOSITY::INFO) std::cout << std::format("Saving NN parameters to {}", root_dir.string()) << std::endl;

	for (auto&& layer : mLayers)
	{
		layer->saveParameters(root_dir, mName);
	}
}

void NeuralNetwork::loadParameters()
{
	std::filesystem::path root_dir = std::filesystem::current_path() / SAVE_DIR_ROOT / mName.c_str();

	if (!std::filesystem::exists(root_dir) || root_dir.empty())
	{
		std::cout << std::format("No parameters to load from {}", root_dir.string()) << std::endl;
		return;
	}

	if (mVerbosity >= VERBOSITY::INFO) std::cout << std::format("Loading {} parameters from {}", mName, root_dir.string()) << std::endl;

	for (auto&& layer : mLayers)
	{
		layer->loadParameters(root_dir, mName);
	}
}
