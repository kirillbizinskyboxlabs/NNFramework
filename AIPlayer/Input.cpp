#include "Input.h"

import <format>;

Input::Input(cudnnHandle_t& handle, int64_t nbDims, int64_t dims[], const Hyperparameters& hyperparameters, bool verbose, std::string name)
	: Layer(handle, nullptr, hyperparameters, verbose, std::move(name))
	, mNbDims(nbDims)
{
	constexpr int64_t alignment = 16;
	constexpr cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
	constexpr cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

	std::vector<int64_t> stride(mNbDims);

	Utils::generateStrides(dims, stride.data(), mNbDims, tensorFormat);

	mOutputTensor = std::make_unique<cudnn_frontend::Tensor>(cudnn_frontend::TensorBuilder()
		.setAlignment(alignment)
		.setDataType(dataType)
		.setDim(mNbDims, dims)
		.setStride(mNbDims, stride.data())
		.setId(generateTensorId())
		.build());

	int64_t size = std::accumulate(dims, dims + mNbDims, 1ll, std::multiplies<int64_t>());
	mOutputSurface = std::make_unique<Surface<float>>(size, 0.0f);

	if (mVerbose)
	{
		std::cout << std::format("Input Layer Tensor\n{}", mOutputTensor->describe()) << std::endl;
	}
}

float* Input::getHostPtr()
{
	return mOutputSurface->hostPtr;
}

void Input::propagateForward()
{
	if (mVerbose)
	{
		std::cout << "Propagating forward on Input" << std::endl;
		//printOutput();
	}

	mOutputSurface->hostToDevSync();
}

void Input::printOutput()
{
	if (!mVerbose)
	{
		return;
	}

	auto dims = mOutputTensor->getDim();
	auto nbDims = mOutputTensor->getDimCount();
	auto stride = mOutputTensor->getStride();

	//for (size_t d = 0; d < nbDims; ++d)
	//{
	//	
	//}

	for (size_t b = 0; b < dims[0]; ++b)
	{
		for (size_t h = 0; h < dims[2]; ++h)
		{
			for (size_t w = 0; w < dims[3]; ++w)
			{
				std::cout << std::format("{} ", mOutputSurface->hostPtr[b * stride[0] + h * stride[2] + w]);
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
	}
}


