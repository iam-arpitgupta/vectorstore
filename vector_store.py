import numpy as np

class VectorStore:
    def __init__(self):
        self.vector_data = {}  #dict to store vector 
        self.vector_index = {} # a dict for index structure for retrievel

    def add_vector(self, vector_id, vector):
        """
        Add a vector to the store 

        Args:
        vector_id (str or int ): A unique id for the vector 
        vector(numpy.darray) : vector data to be stores
        """
        self.vector_data[vector_id] = vector 
        self.update_index(vector_id, vector)
    
    def get_vector(self, vector_id):
        """
        Get a vector from the vector store 
        
        Args:
        vector_id(str or int): a unique id for the vector 

        Returns:
        numpy.darray : the vector data if found , or none if not found 
        """
        return self.vector_data.get(vector_id)
    
    def update_index(self, vector_id,vector):
        """
        updating the indexing structure for the vector store 

        Args: 
        vector_id(str or int): a unique id for the vector 
        vector(numpy.darray) : vector data to be stores
        """

        for exisisting_id , existing_vector in self.vector_data.items():
            similarity = np.dot(vector,existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            if exisisting_id not in self.vector_index:
                self.vector_index[exisisting_id] = {}
            self.vector_index[exisisting_id][vector_id] = similarity
    
    def find_similar_vectors(self,query_vector,num_results = 5):
        """
        find the similar vectors to the query vector
        
        
        Args :
        query_vector (numpy.darray): the query vector 
        num_results (int) : no. of reults to return 
        
        return : 
        list 
        """
        results = []
        for vector_id , vector in self.vector_data.items():
            similarity = np.dot(query_vector,vector) / (np.linalg.norm(vector) * np.linalg.norm(vector))
            results.append((vector_id,similarity))
        #in descending order 
        results.sort(key=lambda x:x[1],reverse=True)
        #top n results
        return results[:num_results]






        