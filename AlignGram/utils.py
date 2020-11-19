import numpy as np
from Bio.SubsMat.MatrixInfo import blosum62
from Bio.pairwise2 import align
from tqdm import tqdm

AMINO_ACIDS = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']


def generateVocabulary(K=3):
	"""
	Generate the k-mer vocabulary

	Args:
		K (int, optional): [k of k-mer]. Defaults to 3.

	Returns:
		tuple[list,dict] : returns a tuple containing vocabulary and vocabulary_mapper
		
							vocabulary (list[str]) : vocabulary contains the list of all possible k-mers
							vocabulary_mapper (dict[str,int]) : vocabulary_mapper maps the k-mers to their
																indices in the vocabulary array
	"""    

	
	vocabulary = ['']

	for i in range(K):

		tmp = []

		for aa in AMINO_ACIDS:
			
			for v in vocabulary:
				
				tmp.append(v+aa)
			
		vocabulary = tmp[:]

	vocabulary_mapper = {}

	for i in range(len(vocabulary)):

		vocabulary_mapper[vocabulary[i]] = i

	return vocabulary,vocabulary_mapper



def computeAlignmentScores(vocabulary, sub_mat, gap_open, gap_extend):
	"""
	Pre-computes the alignment scores between all the k-mer pairs

	Args:
		vocabulary (list [str]): The list of all possible k-mers
		sub_mat (dict[tuple[str,str],int]): Substitution matrix as represented in Bio.SubsMat.MatrixInfo.blosum objects
		gap_open (int): gap open loss (negative number to use Biopython)
		gap_extend (int): gap extend loss (negative number to use Biopython)

	Returns:
		2D numpy array [int,int] : a matrix[len(vocabulary)xlen(vocabulary)] where the value of (i,j) cell presents
								   the alignment score between ith and jth k-mer in the vocabulary
	"""

	alignment_scores = np.zeros((len(vocabulary),len(vocabulary)))

	for i in tqdm(range(len(vocabulary)), desc='Computing Alignment Score'):

		for j in range(i,len(vocabulary)):

			alignment_score = align.globalds(vocabulary[i], vocabulary[j], sub_mat, gap_open, gap_extend, score_only=True)

			alignment_scores[i][j] = alignment_score
			alignment_scores[j][i] = alignment_score

	return alignment_scores


def prepareData(vocabulary,alignment_scores):
	"""
	Prepares the data to train the model    

	Args:
		vocabulary (list [str]): The list of all possible k-mers
		alignment_scores (2D numpy array [int,int]) : a matrix[len(vocabulary)xlen(vocabulary)] where the value of (i,j) cell presents
													  the alignment score between ith and jth k-mer in the vocabulary

	Returns:
		tuple[numpy array,numpy array] : returns a tuple containing input(X) and output(Y) to train the model
	"""

	X = []
	Y = []

	for i in tqdm(range(len(vocabulary)), desc='Preparing Data' ):

		x = np.zeros((len(vocabulary),))
		x[i] = 1.0                                  # one hot encoded values

		y = np.zeros((len(vocabulary),))            

		for j in range(len(vocabulary)):

			y[j] = alignment_scores[i][j]           # alignment scores

		X.append(x)
		Y.append(y)

	X = np.array(X)
	Y = np.array(Y)

	return (X,Y)


