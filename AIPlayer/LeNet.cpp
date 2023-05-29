module;

#include "NeuralNetwork.h"

module LeNet;

import <iostream>;
import <format>;

import MNISTData;

namespace LeNet
{
    void LenetForward()
    {
        std::cout << "LeNet Test v1.0" << std::endl;

        // Hyperparameters?
        constexpr int64_t batchSize = 512;
        constexpr int64_t inputH = 28;
        constexpr int64_t inputW = 28;
        constexpr int64_t C1Features = 32;
        constexpr int64_t C1KernelSize = 5;
        constexpr int64_t C1Padding = 2;
        constexpr int64_t C3Features = 64;
        constexpr int64_t C3KernelSize = 5;
        constexpr int64_t C3Padding = 0;
        constexpr int64_t C5Features = 1024;
        constexpr int64_t C5KernelSize = 5;
        constexpr int64_t C5Padding = 0;
        constexpr int64_t FC6OutputSize = 84;
        constexpr int64_t FC7OutputSize = 10;

        bool verbose = false;

        MNISTDataHolder dh;
        dh.initialize();
        //auto [image, label] = dh.getNextTrain();
        //auto [images, labels] = dh.getNextNTrain(batchSize);
        auto [rows, cols] = dh.getDimensions();

        std::vector<size_t> dims = { 1, rows, cols };

        NeuralNetwork nn(batchSize, dims.size(), dims.data(), NeuralNetwork::VERBOSITY::MIN);
        nn.addConvBiasAct(C1KernelSize, C1Features, C1Padding, verbose, "C1");
        nn.addPool(verbose, "S2");
        nn.addConvBiasAct(C3KernelSize, C3Features, C3Padding, verbose, "C3");
        nn.addPool(verbose, "S4");
        nn.addConvBiasAct(C5KernelSize, C5Features, 0, verbose, "FC5");
        //nn.addConvBiasAct(1, FC6OutputSize, 0, verbose, "FC6");
        nn.addConvBiasAct(1, FC7OutputSize, 0, verbose, "FC7");
        nn.addSoftmax(verbose);
        nn.addCrossEntropy(verbose);


        size_t epoch_num = 200;
        size_t iter_num = epoch_num * (dh.getTrainSize() / batchSize);


        while (iter_num--)
        {
            dh.loadData(batchSize, nn.getInputDataPtr(), nn.getLabelDataPtr());
            nn.syncLabel();

            nn.train();

            if (iter_num % (dh.getTrainSize() / batchSize) == 0)
            {
                std::cout << std::format("Iter {} ", iter_num);
                nn.printLoss();
                nn.printOutput();
            }
        }
    }
}
