"""
Main Driver Code

"""
from progan.proTrain import trainer

import torch

torch.autograd.set_detect_anomaly(True)


def main():
    gan = trainer(
        "cifar-10-images",  # path to data
        128,  # batch size
        [90, 10, 0],  # data split
        "ModelWeights",  # save location
        lr=[0.001, 0.001],  # learning Rates
        merge_samples_const=17  # merge rate
    )
    gan.train(17, 100)
    gan.step_up()  # -> 8,8
    gan.train(17, 100)
    gan.train(17, 100)
    gan.step_up()  # -> 16,16
    gan.train(17, 100)
    gan.train(17, 100)
    gan.step_up()  # -> 32,32
    gan.train(17, 100)
    gan.train(17, 100)
    gan.step_up()  # -> 64,64
    gan.train(17, 100)
    gan.train(17, 100)
    gan.step_up()  # -> 128,128
    gan.train(17, 100)
    gan.train(17, 100)


if __name__ == "__main__":
    main()
