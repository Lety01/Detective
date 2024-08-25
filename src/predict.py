import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Carica il modello salvato
model = tf.keras.models.load_model('models/mnist_cnn_model.h5')

# Carica i dati di test (puoi anche caricare un'immagine esterna)
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
x_test = x_test[..., tf.newaxis]

# Fai una previsione sulla prima immagine del dataset di test
predictions = model.predict(x_test)

# Mostra il risultato per la prima immagine nel set di test
print(f"Prediction: {np.argmax(predictions[0])}, True label: {y_test[0]}")

# Visualizza l'immagine e la previsione
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[0])}, True: {y_test[0]}")
plt.show()
