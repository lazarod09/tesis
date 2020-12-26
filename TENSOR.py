import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import tempfile
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'yeni.csv')
X = data[['X1', 'X2', 'X3', 'X4', 'X5']].values
Y = data['Y'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


#Define the model architecture.
def setup_model():
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10)
    ])
    return model

# Train the digit classification model
def setup_pretrained_weights():
  model= setup_model()

  model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer='adam',
      metrics=['accuracy']
  )

  model.fit(X_train,y_train)

  _, pretrained_weights = tempfile.mkstemp('.tf')

  model.save_weights(pretrained_weights)

  return pretrained_weights

def setup_pretrained_model():
  model = setup_model()
  pretrained_weights = setup_pretrained_weights()
  model.load_weights(pretrained_weights)
  return model

setup_model()
pretrained_weights = setup_pretrained_weights()

base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy

quant_aware_model = tfmot.quantization.keras.quantize_model(base_model)
quant_aware_model.summary()
# Create a base model
base_model = setup_model()
base_model.load_weights(pretrained_weights) # optional but recommended for model accuracy

# Helper function uses `quantize_annotate_layer` to annotate that only the
# Dense layers should be quantized.
def apply_quantization_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer

# Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense`
# to the layers of the model.
annotated_model = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_quantization_to_dense,
)

# Now that the Dense layers are annotated,
# `quantize_apply` actually makes the model quantization aware.
quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
quant_aware_model.summary()
print(base_model.layers[0].name)
