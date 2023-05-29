#pragma once
#include "Layer.h"

class Pool : public Layer
{
public:
	Pool(cudnnHandle_t& handle,
		Layer* previousLayer,
		bool verbose = false,
		std::string name = "Pool");

	void propagateBackward() override;

private:
	void _setupGrad(int64_t poolTensorDim[]);
};

