// AIPlayer.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

import MNISTData;

import LeNet;

import <iostream>;
//import <ranges>;
//import <vector>;
//import <utility>;

void testDataRetrieval();

int main()
{
    LeNet::LenetForward();
}

void testDataRetrieval()
{
    MNISTDataHolder data_holder;
    data_holder.initialize();

    auto [image, label] = data_holder.getNextTrain();

    //std::cout << "Label: " << static_cast<int>(label) << "\n";

    auto display = [](auto image, auto label)
    {
        std::cout << "Label: " << static_cast<int>(label) << "\n";
        for (size_t r = 0; r < 28; ++r)
        {
            for (size_t c = 0; c < 28; ++c)
            {
                std::cout << std::setw(3) << static_cast<int>(image[r * 28 + c]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "\n";
    };

    display(image, label);

    auto [images, labels] = data_holder.getNextNTrain(5);

    std::cout << images.size() << std::endl;

    for (size_t i = 0; i < images.size(); ++i)
    {
        display(images[i], labels[i]);
    }
}
