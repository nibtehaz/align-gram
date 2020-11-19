import pickle
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint

from utils import generateVocabulary, computeAlignmentScores, prepareData
from align_gram_model import AlignGram


def getModel(vocab_len,embedding_len):
	"""
	Creates the Align-gram model

	Args:
		vocab_len (int): number of k-mers
		embedding_len (int): length of embeddings

	Returns:
		keras model: Align-gram model
	"""

	inp = Input(shape=(vocab_len,))

	embedding = Dense(embedding_len, activation='sigmoid')(inp)

	out = Dense(vocab_len, activation='linear')(embedding)

	model = Model(inp, out)

	return model


def trainModel(vocab_len,embedding_len, X, Y, n_epochs=5000):
	"""
	Train the Align gram model

	Args:
		vocab_len (int): number of k-mers
		embedding_len (int): length of embeddings
		X (numpy array): input (one hot encoded)
		Y (numpy array): output (alignment scores)	
		n_epochs (int, optional): number of epochs to train. Defaults to 5000.
	"""

	model = getModel(vocab_len,embedding_len)
	model.compile(optimizer='adam', loss='mse', metrics=['mse'])
	checkpoint_ = ModelCheckpoint('model.h5', verbose=1, monitor='mse',save_best_only=True, mode='min')
	training_history = model.fit(X,Y,epochs=n_epochs,batch_size=64,callbacks=[ checkpoint_])


def buildModel(K, embedding_len, sub_mat, gap_open, gap_extend, model_name=None):
	"""
	Build the Align-gram model

	Args:
		K (int): k of k-mer
		embedding_len (int): length of embeddings
		sub_mat (dict[tuple[str,str],int]): Substitution matrix as represented in Bio.SubsMat.MatrixInfo.blosum objects
		gap_open (int): gap open loss (negative number to use Biopython)
		gap_extend (int): gap extend loss (negative number to use Biopython)
		model_name ([str], optional): Name of the Align-gram model. 
									  Defaults to None, and no model is saved.
									  If a string is used as input the model is saved using that as file name.
	"""

					# preparing data
	vocabulary,vocabulary_mapper = generateVocabulary(K)
	alignment_scores = computeAlignmentScores(vocabulary, sub_mat, gap_open,gap_extend)
	(X,Y) = prepareData(vocabulary,alignment_scores)

					# train the model
	trainModel(len(vocabulary),embedding_len, X, Y)

					# create the align-gram model
	model = getModel(len(vocabulary),embedding_len)
	model.load_weights('model.h5')
	weights = model.layers[1].get_weights()[0]
		
	align_gram = AlignGram(weights,vocabulary, vocabulary_mapper)

	if(model_name != None):
		pickle.dump(align_gram,open('{}.p'.format(model_name),'wb'))

	return align_gram