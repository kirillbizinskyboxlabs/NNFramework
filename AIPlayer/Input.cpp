#include "Input.h"

Input::Input(cudnnHandle_t& handle, int64_t nbDims, int64_t dims[], bool verbose, std::string name)
	: Layer(handle, nullptr, verbose, std::move(name))
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
}

float* Input::getHostPtr()
{
	return mOutputSurface->hostPtr;
}

void Input::propagateForward()
{
	mOutputSurface->hostToDevSync();
}


