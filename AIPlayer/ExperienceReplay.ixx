export module ExperienceReplay;

import <vector>;
import <random>;
import <cassert>;
import <cstring>;
import <filesystem>;

using namespace std;


class State
{
public:
	State(size_t size) noexcept : mData(size) {} // the size of the state should be known ahead of time
	State(State&& other) noexcept
	{
		mData = std::move(other.mData);
	}

	State& operator=(const State& other) = delete;
	State& operator=(State&& other) = delete; // not sure

	virtual ~State() = default;

	const std::vector<float>& getData() const
	{
		return mData;
	}

protected:
	std::vector<float> mData; // storing data in a type negotiated with a NN // TODO: compression/decompression on save/load
};

class SnakeState : public State
{
public:
	SnakeState(size_t size, std::vector<std::vector<char>> frame) noexcept;
};


export class ExperienceReplay
{
public:
	void emplaceExperience(State&& state);

	void loadExperience(float* dataPtr, size_t batchSize) const;

	void saveToDisk(std::filesystem::path experiencePath) const;
	void loadFromDisk(std::filesystem::path experiencePath);

private:
	std::vector<State> mExperience;
};

void ExperienceReplay::emplaceExperience(State&& state)
{
	mExperience.emplace_back(std::move(state));
}

void ExperienceReplay::loadExperience(float* dataPtr, size_t batchSize) const
{
	if (batchSize > mExperience.size())
	{
		throw "Not enough experience to fill the batch";
	}

	std::random_device dev;
	std::mt19937 gen(dev());
	std::uniform_int_distribution dist(0ull, mExperience.size()); // TODO: rethink. This might give duplicated experience. Also, I don't check if we have duplicates when I'm adding new exp. Might be bad.

	for (size_t b = 0; b < batchSize; ++b)
	{
		const auto& instance = mExperience[dist(gen)];
		// TODO: externalize. This should be in the specific to the game providing the experience
		const size_t stateSize = mExperience[0].getData().size(); // this probably should be a const/constexpr set ahead of time ensuring that States have equal shape

		//dataPtr[b * stateSize]
		std::memcpy(dataPtr + b * stateSize, instance.getData().data(), stateSize * sizeof(float));
	}
}

void ExperienceReplay::saveToDisk(std::filesystem::path experiencePath) const
{
	if (mExperiences.empty())
	{
		return;
	}

	if (!std::filesystem::exists(experiencePath))
	{
		std::filesystem::create_directories(experiencePath);
	}

	std::ofstream experienceFile(experiencePath, std::ios::binary);

	experienceFile.write(reinterpret_cast<char*>(&mExperience.size()), sizeof(size_t)); // write number of experiences

	size_t stateSize = mExperience[0].getData().size();

	experienceFile.write(reinterpret_cast<char*>(&stateSize), sizeof(size_t)); // write the size of the state. They expected to be equal.

	for (auto&& exp : mExperience)
	{
		experienceFile.write(reinterpret_cast<char*>(exp.getData().data()), sizeof(float) * stateSize); // TODO: proper data type retrieval
	}
}

void ExperienceReplay::loadFromDisk(std::filesystem::path experiencePath)
{
	if (!std::filesystem::exists(experiencePath))
	{
		return;
	}

	std::ifstream experienceFile(experiencePath, std::ios::binary);

	size_t expSize;

	experienceFile.read(reinterpret_cast<char*>(&expSize), sizeof(size_t)); // read number of stored experiences

	mExperience.reserve(expSize);

	size_t stateSize;

	experienceFile.read(reinterpret_cast<char*>(&stateSize), sizeof(size_t)); // read the size of the state

	for (auto&& exp : mExperience)
	{
		experienceFile.write(reinterpret_cast<char*>(exp.getData().data()), sizeof(float)); // TODO: proper data type retrieval
	}
}

SnakeState::SnakeState(size_t size, std::vector<std::vector<char>> frame) noexcept
	: State(size)
{
	size_t pos = 0;
	assert(frame.size() * frame[0].size() == size);

	for (auto&& line : frame)
	{
		for (auto&& px : line)
		{
			mData[pos++] = static_cast<float>(px) / 255.0f;
		}
	}
}
