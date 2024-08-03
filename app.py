from vector_store import VectorStore
import numpy as np
#import nltk 
#from gensim.models import Word2Vec

#download NlTK's plunkt tokenizer 
#nltk.download('punkt')

#creating instance 
vector_store = VectorStore()

#define your sentence 
sentences = [
    "i am the best",
    "i will be successful"
]

# Tokenization and Vocabulary Creation
vocabulary = set()
for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Assign unique indices to words in the vocabulary
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorization
sentence_vectors = {}
for sentence in sentences:
    tokens = sentence.lower().split()
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1
    sentence_vectors[sentence] = vector
 

#add vectors to the vector store 
for sentence,vectors in tokenize_sentence.items():
    vector_store.add_vector(sentence,vectors)

#serching for the similarity 
query_sentence = "arko is the best person"
query_vector = np.zeroes(len(vocabulary))
query_tokens = query_sentence.lower().split()

for tokens in query_tokens:
    query_vector[word_to_index[tokens]] += 1

similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=2)

# Print similar sentences
print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")








