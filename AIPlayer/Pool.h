#pragma once
#include "Layer.h"

class Pool : public Layer
{
public:
	Pool(cudnnHandle_t& handle,
		Layer* previousLayer,
		const Hyperparameters& hyperparameters,
		bool verbose = false,
		std::string name = "Pool",
		VERBOSITY verbosity = VERBOSITY::MIN);

	void propagateBackward() override;

private:
	void _setupGrad(int64_t poolTensorDim[]);
};

