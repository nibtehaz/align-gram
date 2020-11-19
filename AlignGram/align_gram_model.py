import numpy as np
import functools
from sklearn.metrics.pairwise import cosine_similarity


class AlignGram(object):

    def __init__(self, embedding_matrix, vocabulary, vocabulary_mapper):
        """
        Constructor for AlignGram objects

        Args:
            embedding_matrix (2D numpy array): The matrix containing the embedding vectors
            vocabulary ([type]): [description]
            vocabulary_mapper ([type]): [description]
        """

        self.embedding_matrix = embedding_matrix

        self.magnitudes = np.sum(embedding_matrix*embedding_matrix,axis=1)**0.5

        self.vocabulary = vocabulary

        self.vocabulary_mapper = vocabulary_mapper

    
    def getNearestNeighbors(self,vctr,top_cnt=1):
        """
        Extracts the nearest neighbors of a particular embedding vector

        Args:
            vctr (numpy array): input embedding vector
            top_cnt (int, optional): number of nearest neighbors to extract. Defaults to 1.
        """

        def compare(item1, item2):              # custom sorting function
            if item1[1] > item2[1]:
                return -1
            elif item1[1] < item2[1]:
                return 1
            else:
                return 0

        vctr_magnitude = np.dot(vctr,vctr)**0.5

        distances = np.matmul(self.embedding_matrix,vctr)*(1/self.magnitudes)/(vctr_magnitude)  # computing distances

        distances_sort = np.array(distances)
        distances_sort.sort()

        dist_threshold = distances_sort[-top_cnt]

        neighbor_indices = np.where(distances>=(dist_threshold-1e-6))
        neighbor_similarity = distances[neighbor_indices]

        neighbors = []

        for i in range(len(neighbor_indices[0])):

            neighbors.append([self.vocabulary[neighbor_indices[0][i]],neighbor_similarity[i]])


        neighbors = sorted(neighbors,key=functools.cmp_to_key(compare))

        return neighbors

    def printNearestNeighbors(self,vctr,top_cnt=1):
        """
        Prints the nearest neighbors of a particular embedding vector

        Args:
            vctr (numpy array): input embedding vector
            top_cnt (int, optional): number of nearest neighbors to show. Defaults to 1.
        """

        neighbors = self.getNearestNeighbors(vctr,top_cnt)          # extracts the nearest neighbors

        for i in range(len(neighbors)):

            print(neighbors[i][0],'\t',neighbors[i][1])
    
    def computeCosineSimilarity(self,v1,v2):
        """
        Computes cosine similarity between two vectors

        Args:
            v1 (numpy array): embedding vector 1
            v2 (numpy array): embedding vector 2

        Returns:
            [float]: cosine similarity score
        """

        return cosine_similarity([v1],[v2])

    
    def getEmbeddingVector(self,kmer):
        """
        Extract the embedding vector for a k-mer

        Args:
            kmer (str): input k-mer

        Returns:
            [numpy array]: embedding vector of the k-mer
        """

        return self.embedding_matrix[self.vocabulary_mapper[kmer]]

    