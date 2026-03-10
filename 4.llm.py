# Install required package
# pip install gensim

# Import required libraries
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
import re

# Step 1: Read corpus from file
with open("cricket.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

# Remove punctuation
text = re.sub(r"[^a-zA-Z\s]", "", text)

# Step 2: Split text into sentences
sentences = re.split(r'\n+', text)

# Remove empty sentences
sentences = [s.strip() for s in sentences if s.strip() != ""]

print("Sentences from file:")
print(sentences)

# Step 3: Tokenize sentences
statements_listt = [sentence.split() for sentence in sentences]
print("\nAfter split statements_listt:", statements_listt)

# Step 4: Remove stopwords
documents = [[word for word in doc if word not in STOPWORDS] for doc in statements_listt]

print("\nCleaned Documents:")
print(documents)

# Step 5: Train Word2Vec model
model = Word2Vec(documents, vector_size=5, window=3, min_count=1)

# Step 6: Vector operations
if all(word in model.wv for word in ['playing', 'winning', 'defending']):
    
    vector1 = model.wv['playing']
    vector2 = model.wv['winning']
    vector3 = model.wv['defending']

    print("\nVector for 'playing':", vector1)
    print("Vector for 'winning':", vector2)
    print("Vector for 'defending':", vector3)

    sum_vector = vector1 + vector2
    diff_vector = vector2 - vector3

    print("\nSum vector (playing + winning):", sum_vector)
    print("Difference vector (winning - defending):", diff_vector)

else:
    print("\nRequired words not found in vocabulary")

# Step 7: Cosine similarity
if 'tournament' in model.wv and 'india' in model.wv:
    similarity = model.wv.similarity('tournament', 'india')
    print("\nCosine Similarity between 'tournament' and 'india':", similarity)

# Step 8: Most similar words
if 'tournament' in model.wv:
    similar_words = model.wv.most_similar('tournament', topn=5)
    print("\nMost Similar words to 'tournament':", similar_words)

# Step 9: Analogy example
if all(word in model.wv for word in ['tournament', 'india', 'players']):
    
    analogy_vector = model.wv['tournament'] - model.wv['india'] + model.wv['players']
    most_similar = model.wv.most_similar(positive=[analogy_vector], topn=1)

    print("\nAnalogy Result (tournament - india + players):", most_similar)
else:
    print("\nAnalogy words not present in vocabulary")