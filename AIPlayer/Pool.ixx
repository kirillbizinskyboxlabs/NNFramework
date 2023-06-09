module;

#include <cudnn_frontend.h>

export module NeuralNetwork:Pool;

import :Layer;

export class Pool : public Layer
{
public:
	Pool(cudnnHandle_t& handle,
		Layer* previousLayer,
		const Hyperparameters& hyperparameters,
		std::string name = "Pool",
		VERBOSITY verbosity = VERBOSITY::MIN,
		const int64_t windowSize = 2,
		const int64_t stride = 2,
		const int64_t pad = 0);

	void propagateBackward() override;

private:
	void _setupGrad();

	const int64_t mNbSpatialDims = 2;
	const float mAlpha = 1.0f;
	const float mBeta = 0.0f;

	//const int64_t mAlignment = 16;
	//const cudnnTensorFormat_t mTensorFormat = CUDNN_TENSOR_NHWC;

	const std::vector<int64_t> mWindowDim;
	const std::vector<int64_t> mPrePadding;
	const std::vector<int64_t> mPostPadding;
	const std::vector<int64_t> mStride;
	const cudnnDataType_t mDataType = CUDNN_DATA_FLOAT;
	const cudnnNanPropagation_t mNanOpt = CUDNN_PROPAGATE_NAN;
	const cudnn_frontend::cudnnResampleMode_t mPoolMode = cudnn_frontend::cudnnResampleMode_t::CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
	const cudnn_frontend::cudnnPaddingMode_t mPaddingMode = cudnn_frontend::cudnnPaddingMode_t::CUDNN_ZERO_PAD;
};
