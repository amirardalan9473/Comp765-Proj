from keras.layers import Input, Dense
from keras.models import Model


def autoencod(x_train, x_test, ):
    encoding_dim = 12  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
    input_img = Input(shape=(4,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(4, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

    encoded_input = Input(shape=(encoding_dim,))

    decoder_layer = autoencoder.layers[-1]

    decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # encoded_imgs = encoder.predict(x_test)
    # decoded_imgs = decoder.predict(encoded_imgs)
    decoder.save('decoder.h5')
    encoder.save('encoder.h5')
    print("Saved model to disk")


if __name__ == "__main__":
    from pend_exp import sampler


