import tensorflow as tf

def get_model_MPG_FF(seed):

    initializer = tf.keras.initializers.GlorotNormal(seed)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer, input_shape=[9]))
    model.add(tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

    opt = tf.keras.optimizers.Adam(0.001)
    mtrc = tf.keras.metrics.mse
    model.compile(loss="mse", optimizer=opt, metrics=[mtrc])
    return model


def get_model_IRIS_FF(seed):

    initializer = tf.keras.initializers.GlorotNormal(seed)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer, input_shape=[4]))
    model.add(tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(3, activation='softmax', kernel_initializer=initializer))

    opt = tf.keras.optimizers.Adam(0.001)
    mtrc = tf.keras.metrics.categorical_accuracy
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=[mtrc])
    return model


def get_model_MNIST_CONV(seed):

    initializer = tf.keras.initializers.GlorotNormal(seed)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(20, (3, 3), activation='tanh', kernel_initializer = initializer, input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.AveragePooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(20, (3, 3), activation='tanh', kernel_initializer = initializer))
    model.add(tf.keras.layers.AveragePooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='tanh', kernel_initializer = initializer))
    model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer))


    opt = tf.keras.optimizers.Adam(0.001)
    mtrc = tf.keras.metrics.categorical_accuracy
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=[mtrc])
    return model

def get_model_MNIST_FF(seed):

    initializer = tf.keras.initializers.GlorotNormal(seed)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
    model.add(tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(100, activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer))


    opt = tf.keras.optimizers.Adam(0.001)
    mtrc = tf.keras.metrics.categorical_accuracy
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=[mtrc])
    return model

def get_model_CIFAR10_CONV(seed):

    initializer = tf.keras.initializers.GlorotNormal(seed)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(60, (3, 3), activation='tanh', kernel_initializer=initializer, input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.AveragePooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(60, (3, 3), activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.AveragePooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(300, activation='tanh', kernel_initializer=initializer))
    model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer))

    opt = tf.keras.optimizers.Adam(0.001)
    mtrc = tf.keras.metrics.categorical_accuracy
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=[mtrc])
    return model


if __name__=="__main__":
    pass