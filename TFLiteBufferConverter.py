
import tensorflow as tf

tf.logging.set_verbosity( tf.logging.ERROR )

keras_model_path = input( "Enter Keras model file path > " )

converter = tf.lite.TFLiteConverter.from_keras_model_file( keras_model_path )
converter.post_training_quantize = True
tflite_buffer = converter.convert()
open( 'android/model.tflite' , 'wb' ).write( tflite_buffer )

print( 'TFLite model created.')