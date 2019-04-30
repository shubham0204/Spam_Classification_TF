
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv( 'raw_data/data.csv' , encoding='iso8859' , usecols=[ 'v1' , 'v2' ] )

labels = list()
for line in dataframe.v1:
	labels.append( 0 if line == 'ham' else 1 )
texts = list()
for line in dataframe.v2:
	texts.append( line )
lengths = list()
for text in texts:
	lengths.append( len( text.split() ) )
maxlen = max( lengths )
labels = np.array( labels )
texts = np.array( texts )

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts( texts )
tokenized_messages = tokenizer.texts_to_sequences( texts )
padded_messages = tf.keras.preprocessing.sequence.pad_sequences( tokenized_messages , maxlen )
onehot_labels = tf.keras.utils.to_categorical( labels , num_classes=2 )

X = padded_messages
Y = onehot_labels

print( X.shape )
print( Y.shape )
print( 'MESSAGE MAXLEN = {}'.format( maxlen ) )

train_features , test_features ,train_labels, test_labels = train_test_split( X , Y , test_size=0.4 )
output_path = 'processed_data/'
np.save( '{}x.npy'.format( output_path )  , train_features )
np.save( '{}y.npy'.format( output_path )  , train_labels )
np.save( '{}test_x.npy'.format( output_path ) , test_features )
np.save( '{}test_y.npy'.format( output_path ) , test_labels )

with open( 'android/word_dict.json' , 'w' ) as file:
	json.dump( tokenizer.word_index , file )

print( 'Data processed.')


