#!/usr/bin/env python3
'''
trains a convolutional neural network to classify the CIFAR 10 dataset
'''

import tensorflow.keras as K


def preprocess_data(X, Y):
    '''
    pre-processes the data for the model
    :X: is a numpy.ndarray of shape (m, 32, 32, 3) containing the
    CIFAR 10 data, where m is the number of data points
    :Y: is a numpy.ndarray of shape (m,) containing the CIFAR 10
    labels for X
    '''
    X_p = K.applications.resnet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


if __name__ == '__main__':

    # Data onboarding
    # Load the cifar10 dataset and split dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    # preprocess the data
    X_train_p, Y_train_p = preprocess_data(x_train, y_train)
    X_test_p, Y_test_p = preprocess_data(x_test, y_test)

    # BUILDING THE MODEL 
    resnet_base = K.applications.ResNet50(
        include_top=False,
        input_shape=(224, 224, 3)
    )

    input_layer = K.Input(shape=(32, 32, 3))
    resizing_layer = K.layers.Lambda(
        lambda img:
        # Resize images to a target size without aspect ratio distortion.
        K.preprocessing.image.smart_resize(
            img, (224, 224)))(input_layer)

    resnet_layer = resnet_base(resizing_layer, training=False)
    flatten_layer = K.layers.Flatten()(resnet_layer)
    d1_layer = K.layers.Dense(500, activation='relu')(flatten_layer)
    dropout_layer = K.layers.Dropout(0.3)(d1_layer)
    output_layer =  K.layers.Dense(10, activation='softmax')(dropout_layer)
    model = K.Model(inputs=input_layer, outputs=output_layer)

    # check the model summary
    model.summary()

    # Training model
    # Freeze the base network
    resnet_base.trainable = False

    model.compile(K.optimizers.Adam(learning_rate=.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    history = model.fit(
            X_train_p,
            Y_train_p,
            validation_data=(X_test_p, Y_test_p),
            batch_size=300,
            epochs=4,
            verbose=1)

    results = model.evaluate(X_test_p, Y_test_p)

    # save model
    model.save('cifar10.h5')
