
from tensorflow.keras import models , optimizers , losses ,activations
from tensorflow.keras.layers import *
import tensorflow as tf
import time

class Classifier (object) :

	def __init__( self , number_of_classes , maxlen ):
		dropout_rate = 0.5
		input_shape = ( maxlen ,  )
		target_shape = ( maxlen , 1 )
		self.model_scheme = [
			Reshape( input_shape=input_shape , target_shape=( maxlen , 1 ) ),
			Conv1D( 128, kernel_size=2 , strides=1, activation=activations.relu , kernel_regularizer='l1'),
			MaxPooling1D(pool_size=2 ),
			Flatten() ,
			Dense( 64 , activation=activations.relu ) ,
			BatchNormalization(),
			Dropout(dropout_rate),
			Dense( number_of_classes, activation=tf.nn.softmax )
		]

		self.__model = tf.keras.Sequential(self.model_scheme)
		self.__model.compile(
			optimizer=optimizers.Adam( lr=0.0001 ),
			loss=losses.categorical_crossentropy ,
			metrics=[ 'accuracy' ] ,
		)


	def fit(self, X, Y ,  hyperparameters  ):
		initial_time = time.time()
		self.__model.fit( X  , Y ,
						 batch_size=hyperparameters[ 'batch_size' ] ,
						 epochs=hyperparameters[ 'epochs' ] ,
						 callbacks=hyperparameters[ 'callbacks'],
						 validation_data=hyperparameters[ 'val_data' ]
						 )
		final_time = time.time()
		eta = ( final_time - initial_time )
		time_unit = 'seconds'
		if eta >= 60 :
			eta = eta / 60
			time_unit = 'minutes'
		self.__model.summary( )
		print( 'Elapsed time acquired for {} epoch(s) -> {} {}'.format( hyperparameters[ 'epochs' ] , eta , time_unit ) )


	def evaluate(self , test_X , test_Y  ) :
		return self.__model.evaluate(test_X, test_Y)


	def predict(self, X  ):
		predictions = self.__model.predict( X  )
		return predictions


	def save_model(self , file_path ):
		self.__model.save(file_path )

	def load_model(self , file_path ):
		self.__model = models.load_model(file_path)
