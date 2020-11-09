import numpy as np

class Config_IRIS_FF:
    NAME_EXP = 'IRIS_FF'
    NUM_EXP_TO_RUN = 15
    EXP_INDEX_START = None

    NUM_EPOCHS = 500
    RANGE_REG_PENALTY = sorted(np.logspace(-6, 6, num=13, endpoint=True).tolist() + [0.0])

class Config_MPG_FF:
    NAME_EXP = 'MPG_FF'
    NUM_EXP_TO_RUN = 15
    EXP_INDEX_START = None

    NUM_EPOCHS = 500
    RANGE_REG_PENALTY = sorted(np.logspace(-6, 6, num=13, endpoint=True).tolist() + [0.0])

class Config_MNIST_CONV:
    NAME_EXP = 'MNIST_CONV'
    NUM_EXP_TO_RUN = 15
    EXP_INDEX_START = None

    NUM_EPOCHS = 200
    RANGE_REG_PENALTY = sorted(np.logspace(-6, 6, num=13, endpoint=True).tolist() + [0.0])

class Config_MNIST_FF:
    NAME_EXP = 'MNIST_FF'
    NUM_EXP_TO_RUN = 15
    EXP_INDEX_START = None

    NUM_EPOCHS = 200
    RANGE_REG_PENALTY = sorted(np.logspace(-6, 6, num=13, endpoint=True).tolist() + [0.0])

class Config_CIFAR10_CONV:
    NAME_EXP = 'CIFAR10_CONV'
    NUM_EXP_TO_RUN = 15
    EXP_INDEX_START = None

    NUM_EPOCHS = 200
    RANGE_REG_PENALTY = sorted(np.logspace(-6, 6, num=13, endpoint=True).tolist() + [0.0])