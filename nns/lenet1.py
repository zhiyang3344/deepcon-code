import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import data_util
from keras.utils.vis_utils import plot_model


def tran_and_save_model(to_save_model_file='../models/lenet1-relu'):
    batch_size = 256
    epochs = 10
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    (x_train, y_train), (x_test, y_test) = data_util.get_mnist_data()

    inputs = Input(shape=(28, 28, 1))
    intermdia_layer = Conv2D(4, (5, 5), activation='relu', padding='same', input_shape=input_shape)(inputs)
    intermdia_layer = MaxPooling2D(pool_size=(2, 2))(intermdia_layer)

    intermdia_layer = Conv2D(12, (5, 5), activation='relu', padding='same')(intermdia_layer)
    intermdia_layer = MaxPooling2D(pool_size=(2, 2))(intermdia_layer)

    intermdia_layer = Flatten()(intermdia_layer)
    predictions = Dense(10, activation='softmax')(intermdia_layer)
    model = Model(inputs=inputs, outputs=predictions)
    model.name = 'lenet1'

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)
    model.save(to_save_model_file)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])


if __name__ == '__main__':
    to_save_model_file = '../models/lenet1-relu'
    tran_and_save_model(to_save_model_file)
    lenet1_model = load_model(to_save_model_file)
    lenet1_model.summary()
    plot_model(lenet1_model, to_file='../outputs/model_visualization/lenet1-relu.png', show_shapes=True)
