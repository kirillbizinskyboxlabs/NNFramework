module;

#include "NeuralNetwork.h"

module LeNet;

import <iostream>;
import <format>;
import <cmath>;

import MNISTData;

namespace LeNet
{
    void Lenet(bool train, bool verbose)
    {
        std::cout << "LeNet Test v1.3" << std::endl;

        // Hyperparameters?
        constexpr int64_t batchSize = 128;
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
        constexpr int64_t FC7KernelSize = 1;
        constexpr int64_t FC7Padding = 0;
        constexpr int64_t FC7OutputSize = 10;

        //bool verbose = false;

        VERBOSITY verbosity = verbose ? VERBOSITY::REACH_INFO : VERBOSITY::MIN;

        MNISTDataHolder dh;
        dh.initialize();
        //dh.presentData();
        //auto [image, label] = dh.getNextTrain();
        //auto [images, labels] = dh.getNextNTrain(batchSize);
        auto [rows, cols] = dh.getDimensions();

        std::vector<size_t> dims = { 1, rows, cols };

        NeuralNetwork nn(batchSize, dims.size(), dims.data(), verbosity, "LeNet");
        nn.mHyperparameters.updateType = Hyperparameters::UpdateType::mSGD;

        nn.addConvBiasAct(C1KernelSize, C1Features, C1Padding, "C1");
        nn.addPool("S2");
        nn.addConvBiasAct(C3KernelSize, C3Features, C3Padding, "C3");
        nn.addPool("S4");
        nn.addConvBiasAct(C5KernelSize, C5Features, C5Padding, "FC5");
        //nn.addConvBiasAct(1, FC6OutputSize, 0, "FC6");
        nn.addConvBiasAct(FC7KernelSize, FC7OutputSize, FC7Padding, "FC7");
        nn.addSoftmax();
        nn.addCrossEntropy();


        float minLoss = FLT_MAX;
        size_t minLossIter = 0;
        size_t validationSize = dh.getValidationSize();

        auto validate = [&nn, &dh, validationSize, &minLoss, &minLossIter, batchSize](size_t iter)
        {
            float tmpLoss = 0;
            const size_t validation_iter_num = validationSize / batchSize;
            size_t validation_iter = validation_iter_num;
            while (validation_iter--)
            {
                dh.loadValidationData<float>(batchSize, nn.getInputDataPtr(), nn.getLabelDataPtr());
                nn.syncLabel();

                nn.inference();
                tmpLoss += nn.getLoss(); // avg over batch size
            }

            tmpLoss /= validation_iter_num;

            std::cout << std::format("Loss over validation: {}", tmpLoss) << std::endl;

            if (tmpLoss < minLoss)
            {
                minLoss = tmpLoss;
                minLossIter = iter;
            }
        };

        if (train)
        {
            constexpr size_t epoch_num = 50;
            const size_t epoch_iter = dh.getTrainSize() / batchSize;
            const size_t total_iter = epoch_num * epoch_iter;
            constexpr size_t showPerEpoch = 1;
            size_t frequency = epoch_iter / showPerEpoch;
            constexpr size_t output_frequency = 5;
            size_t iter = 0;
            //size_t frequency = 100;
            //iter_num = 2;


            while (iter < total_iter)
            {
                dh.loadTrainData<float>(batchSize, nn.getInputDataPtr(), nn.getLabelDataPtr());
                nn.syncLabel();

                nn.train();

                if (iter % epoch_iter == 0)
                {
                    std::cout << std::format("Iter {} ", iter);
                    nn.printLoss();

                    if (iter % (epoch_iter * output_frequency) == 0)
                    {
                        nn.printOutput();
                    }

                    validate(iter);

                }

                iter++;
            }

            //nn.saveParameters(); // TODO: make conditional?
        }
        else
        {
            nn.loadParameters();
            validate(0);
        }

        std::cout << std::format("Lowest loss over validation {} on {} iter", minLoss, minLossIter) << std::endl;
        std::cout << std::format("{}% Accuracy", exp(-minLoss) * 100);
    }
}
