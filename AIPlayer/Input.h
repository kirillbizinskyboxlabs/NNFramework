#pragma once
#include "Layer.h"

import <ranges>;

class Input : public Layer
{
public:
	Input(cudnnHandle_t& handle,
		int64_t nbDims,
		int64_t dims[],
		bool verbose = false,
		std::string name = "Input");

	float* getHostPtr(); // loading up data is a responsibility of the user

	void propagateForward() override;
	void propagateBackward() override {} // do nothing

private:
	const int64_t mNbDims;
};
