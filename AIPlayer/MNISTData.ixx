export module MNISTData;

import <filesystem>;
import <iostream>;
import <fstream>;
import <vector>;
import <format>;
import <utility>;
import <ranges>;
import <cassert>;

constexpr char DATA_ROOT[] = "data";
constexpr char TRAIN_IMAGES[] = "train-images.idx3-ubyte";
constexpr char TRAIN_LABELS[] = "train-labels.idx1-ubyte";
constexpr char VALIDATION_IMAGES[] = "t10k-images.idx3-ubyte";
constexpr char VALIDATION_LABELS[] = "t10k-labels.idx1-ubyte";

export class MNISTDataHolder
{
public:
    MNISTDataHolder() = default;

    void initialize();

    //std::pair<std::ranges::ref_view<std::vector<uint8_t>>, uint8_t> getNext() const
    auto getNextTrain()
    {
        if (current >= num_images)
        {
            current = 0;
        }

        auto res = std::make_pair(std::views::all(images[current]), labels[current]);
        ++current;
        return res;
    }

    auto getNextNTrain(uint32_t n)
    {
        if (n > num_images)
        {
            std::cerr << "Too large request\n";
            throw;
        }

        if (current + n >= num_images)
        {
            current = 0;
        }

        auto res = std::make_pair(std::views::counted(images.begin() + current, n), std::views::counted(labels.begin() + current, n));
        current += n;
        return res;
    }

    auto getDimensions() const
    {
        return std::make_pair(num_rows, num_cols);
    }

    template<typename T>
    void loadTrainData(uint32_t batchSize, T* dataPtr, T* labelPtr)
    {
        if (batchSize > num_images)
        {
            std::cerr << "Too large request\n";
            throw;
        }

        // if batchSize isn't a multiply of num_images this might lead to some of them never being used. TODO: rethink
        if (current + batchSize >= num_images)
        {
            current = 0;
        }

        for (uint32_t i = 0; i < batchSize; ++i)
        {
            const uint32_t imageSize = num_rows * num_cols;
            for (uint32_t j = 0; j < imageSize; ++j)
            {
                size_t px = imageSize * i + j;
                dataPtr[px] = (static_cast<T>(images[current + i][j])) / 255.0f;
            }

            constexpr size_t numClasses = 10; // can be member var
            for (size_t j = 0; j < numClasses; ++j)
            {
                if (labels[current + i] == j)
                {
                    labelPtr[i * numClasses + j] = 1;
                }
                else
                {
                    labelPtr[i * numClasses + j] = 0;
                }
            }
        }

        current += batchSize;
    }

    template<typename T>
    void loadValidationData(uint32_t batchSize, T* dataPtr, T* labelPtr)
    {
        if (batchSize > mValidationImages.size())
        {
            std::cerr << "Too large request\n";
            throw;
        }

        if (mCurrentValidation + batchSize >= mValidationImages.size())
        {
            mCurrentValidation = 0;
        }

        for (uint32_t i = 0; i < batchSize; ++i)
        {
            const uint32_t imageSize = num_rows * num_cols;
            for (uint32_t j = 0; j < imageSize; ++j)
            {
                size_t px = imageSize * i + j;
                dataPtr[px] = (static_cast<T>(mValidationImages[mCurrentValidation + i][j])) / 255.0f;
            }

            constexpr size_t numClasses = 10; // can be member var
            for (size_t j = 0; j < numClasses; ++j)
            {
                if (mValidationLabels[mCurrentValidation + i] == j)
                {
                    labelPtr[i * numClasses + j] = 1;
                }
                else
                {
                    labelPtr[i * numClasses + j] = 0;
                }
            }
        }

        mCurrentValidation += batchSize;
    }

    auto getTrainSize() const
    {
        return images.size();
    }

    auto getValidationSize() const
    {
        return mValidationImages.size();
    }

    void presentData() const;

private:
    static void _loadImagesAndLabels(std::vector<std::vector<uint8_t>>& images, 
                                     std::vector<uint8_t>& labels, 
                                     uint32_t& numImages, 
                                     uint32_t& numRows, 
                                     uint32_t& numCols,
                                     const std::filesystem::path& image_path,
                                     const std::filesystem::path& label_path);

    std::vector<std::vector<uint8_t>> images;
    std::vector<uint8_t> labels;

    uint32_t current = 0;

    std::vector<std::vector<uint8_t>> mValidationImages;
    std::vector<uint8_t> mValidationLabels;

    uint32_t mCurrentValidation = 0;

    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_cols;
};

void MNISTDataHolder::initialize()
{
    namespace fs = std::filesystem;

    fs::path current_dir = std::filesystem::current_path();

    // Define the path to the MNIST dataset directory
    fs::path data_dir = current_dir / DATA_ROOT;

    // Define the paths to the image and label files
    fs::path image_path = data_dir / TRAIN_IMAGES;
    fs::path label_path = data_dir / TRAIN_LABELS;
    
    _loadImagesAndLabels(images, labels, num_images, num_rows, num_cols, image_path, label_path);

    image_path = data_dir / VALIDATION_IMAGES;
    label_path = data_dir / VALIDATION_LABELS;

    uint32_t num_validation_images;
    uint32_t num_validation_rows;
    uint32_t num_validation_cols;

    _loadImagesAndLabels(mValidationImages, mValidationLabels, num_validation_images, num_validation_rows, num_validation_cols, image_path, label_path);

    assert(num_validation_rows == num_rows); // sanity check
    assert(num_validation_cols == num_cols);
}

void MNISTDataHolder::presentData() const
{
    auto printRow = [num_cols = num_cols](size_t row, const std::vector<uint8_t>& image)
    {
        for (size_t c = 0; c < num_cols; ++c)
        {
            std::cout << std::setw(3) << static_cast<int>(image[row * num_cols + c]) << " ";
        }
    };

    constexpr char spacer[] = "      ";

    for (size_t r = 0; r < num_rows; ++r)
    {
        printRow(r, images[0]);
        std::cout << spacer;
        printRow(r, images[1]);
        std::cout << std::endl;
    }
}

void MNISTDataHolder::_loadImagesAndLabels(std::vector<std::vector<uint8_t>>& images,
                                           std::vector<uint8_t>& labels, 
                                           uint32_t& num_images,
                                           uint32_t& num_rows,
                                           uint32_t& num_cols,
                                           const std::filesystem::path& image_path,
                                           const std::filesystem::path& label_path)
{
    // Open the image file
    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file) {
        std::cerr << "Error: Could not open image file\n";
        return;
    }

    // Open the label file
    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file) {
        std::cerr << "Error: Could not open label file\n";
        return;
    }

    // Read the header of the image file
    uint32_t magic;
    image_file.read(reinterpret_cast<char*>(&magic), 4);
    image_file.read(reinterpret_cast<char*>(&num_images), 4);
    image_file.read(reinterpret_cast<char*>(&num_rows), 4);
    image_file.read(reinterpret_cast<char*>(&num_cols), 4);
    magic = _byteswap_ulong(magic);
    num_images = _byteswap_ulong(num_images);
    num_rows = _byteswap_ulong(num_rows);
    num_cols = _byteswap_ulong(num_cols);

    std::cout << std::format("{} {} {}", num_images, num_rows, num_cols) << std::endl;

    // Verify that the file format is correct
    if (magic != 0x00000803) {
        std::cerr << "Error: Incorrect file format\n";
        return;
    }

    // Read the header of the label file
    uint32_t num_labels;
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    magic = _byteswap_ulong(magic);
    num_labels = _byteswap_ulong(num_labels);

    // Verify that the file format is correct
    if (magic != 0x00000801) {
        std::cerr << "Error: Incorrect file format\n";
        return;
    }

    // Verify that the file format is correct
    if (num_labels != num_images) {
        std::cerr << "Error: num_labels != num_images\n";
        return;
    }
    images.resize(num_images, std::vector<uint8_t>(num_rows * num_cols));
    labels.resize(num_labels);

    for (uint32_t i = 0; i < num_images; ++i)
    {
        image_file.read(reinterpret_cast<char*>(images[i].data()), num_rows * num_cols);
        label_file.read(reinterpret_cast<char*>(&labels[i]), 1);
    }
}
