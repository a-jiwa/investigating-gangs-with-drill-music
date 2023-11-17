import os
import pandas as pd
import nltk
import string
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set the folder path containing your CSV files
folder_path = './lyrics'

# New folder path for cleaned lyrics
cleaned_folder_path = './cleaned_lyrics'

# Create the cleaned_lyrics folder if it doesn't exist
if not os.path.exists(cleaned_folder_path):
    os.makedirs(cleaned_folder_path)

# Get a list of stop words
stop_words = set(stopwords.words('english'))

# Function to clean, remove punctuation, and tokenize lyrics
def clean_and_tokenize_lyrics(lyrics):
    # Remove punctuation
    lyrics = lyrics.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    lyrics = lyrics.lower()
    # Tokenize the lyrics
    tokens = word_tokenize(lyrics)
    # Remove stop words
    filtered_lyrics = [word for word in tokens if not word in stop_words]
    return filtered_lyrics

# Process each CSV file in the folder
for file in os.listdir(folder_path):
    if file.endswith('_lyrics.csv'):
        file_path = os.path.join(folder_path, file)
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Apply the cleaning and tokenizing function to the lyrics column
        df['cleaned_lyrics'] = df['Lyrics'].apply(clean_and_tokenize_lyrics)
        # Save the modified dataframe to the new folder
        df.to_csv(os.path.join(cleaned_folder_path, file), index=False)

# Initialize a list to store the corpus and song titles
corpus = []
song_titles = []

# Process each cleaned CSV file
for file in os.listdir(cleaned_folder_path):
    if file.endswith('_lyrics.csv'):
        file_path = os.path.join(cleaned_folder_path, file)
        df = pd.read_csv(file_path)

        # Assuming each CSV file has a 'Song Title' column
        for _, row in df.iterrows():
            lyrics_list = row['cleaned_lyrics'].strip("[]").replace("'", "").split(", ")
            corpus.append(lyrics_list)
            song_titles.append(row['Title'])  # Collect the corresponding song title

# Create a dictionary from the corpus
dictionary = corpora.Dictionary(corpus)

# Convert corpus into Document-Term Matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in corpus]

# Applying the PLSA model using LDA
lda_model = models.LdaModel(doc_term_matrix, num_topics=10, id2word=dictionary, passes=100)

# Create a Document-Topic Matrix with Zero Padding for Equal Length Vectors
num_topics = 10
doc_topic_matrix_padded = []
doc_topic_matrix = lda_model.get_document_topics(doc_term_matrix, minimum_probability=0)
for doc in doc_topic_matrix:
    topic_dict = dict(doc)
    padded_vector = [topic_dict.get(i, 0) for i in range(num_topics)]
    doc_topic_matrix_padded.append(padded_vector)

# Calculate cosine similarity for each pair of documents
similarity_matrix = []
for doc1 in doc_topic_matrix_padded:
    similarities = []
    for doc2 in doc_topic_matrix_padded:
        sim = cosine_similarity([doc1, doc2])
        similarities.append(sim[0, 1])
    similarity_matrix.append(similarities)

# Create JSON data for song similarity
json_data_songs = {
    "nodes": [{"id": title} for title in song_titles],
    "links": [
        {"source": song_titles[i],
         "target": song_titles[j],
         "value": float(similarity_matrix[i][j])}
        for i in range(len(similarity_matrix))
        for j in range(i+1, len(similarity_matrix))
        if similarity_matrix[i][j] > 0.05  # Adjust threshold as needed
    ]
}

# Output the JSON data to a file
with open('song_similarity_graph.json', 'w') as json_file:
    json.dump(json_data_songs, json_file)

print("JSON data for song similarity visualization saved as 'song_similarity_graph.json'")

# Creating a network graph
G = nx.Graph()

# Add nodes (each node is a song)
for i in range(len(similarity_matrix)):
    G.add_node(i)

# Add edges (only if similarity is above a certain threshold, e.g., 0.5)
for i in range(len(similarity_matrix)):
    for j in range(len(similarity_matrix)):
        if i != j and similarity_matrix[i][j] > 0.05:
            print('>')
            G.add_edge(i, j, weight=similarity_matrix[i][j])

# Draw the network graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5)  # k regulates the distance between nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', linewidths=1, font_size=10)
plt.show()


# Similarity of artists
# Initialize a dictionary to store aggregated lyrics by artist
artist_lyrics = {}

# Process each cleaned CSV file
for file in os.listdir(cleaned_folder_path):
    if file.endswith('_lyrics.csv'):
        artist_name = file.replace('_lyrics.csv', '')
        file_path = os.path.join(cleaned_folder_path, file)
        df = pd.read_csv(file_path)
        aggregated_lyrics = ' '.join(df['cleaned_lyrics'].apply(lambda x: ' '.join(eval(x))))
        artist_lyrics[artist_name] = aggregated_lyrics

# Create a new corpus with aggregated artist lyrics
artist_corpus = [artist_lyrics[artist].split() for artist in artist_lyrics]

# Create a dictionary from the artist_corpus
dictionary = corpora.Dictionary(artist_corpus)

# Convert corpus into Document-Term Matrix
doc_term_matrix = [dictionary.doc2bow(doc) for doc in artist_corpus]

# Applying the PLSA model using LDA
lda_model = models.LdaModel(doc_term_matrix, num_topics=10, id2word=dictionary, passes=100)

# Create a Document-Topic Matrix with Zero Padding for Equal Length Vectors
num_topics = 10
doc_topic_matrix_padded = []

doc_topic_matrix = lda_model.get_document_topics(doc_term_matrix, minimum_probability=0)
for doc in doc_topic_matrix:
    topic_dict = dict(doc)
    padded_vector = [topic_dict.get(i, 0) for i in range(num_topics)]
    doc_topic_matrix_padded.append(padded_vector)

# Initialize an empty list to store similarity values
similarity_matrix = []

# Calculate cosine similarity for each pair of artists
for doc1 in doc_topic_matrix_padded:
    similarities = []
    for doc2 in doc_topic_matrix_padded:
        sim = cosine_similarity([doc1, doc2])
        similarities.append(sim[0, 1])
    similarity_matrix.append(similarities)

# Creating a network graph
G = nx.Graph()

# Add nodes (each node is an artist)
for i, artist in enumerate(artist_lyrics.keys()):
    G.add_node(artist)

# Add edges (only if similarity is above a certain threshold, e.g., 0.2)
threshold = 0.2  # Adjust this value as needed
for i in range(len(similarity_matrix)):
    for j in range(len(similarity_matrix)):
        if i != j and similarity_matrix[i][j] > threshold:
            G.add_edge(list(artist_lyrics.keys())[i], list(artist_lyrics.keys())[j], weight=similarity_matrix[i][j])

# Draw the network graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.5)  # k regulates the distance between nodes
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', linewidths=1, font_size=10)
plt.show()


# Create JSON data for D3.js visualization
json_data = {
    "nodes": [{"id": artist} for artist in artist_lyrics.keys()],
    "links": [
        {"source": list(artist_lyrics.keys())[i],
         "target": list(artist_lyrics.keys())[j],
         "value": float(similarity_matrix[i][j])}  # Convert to standard float
        for i in range(len(similarity_matrix))
        for j in range(i+1, len(similarity_matrix))  # Avoid duplicate edges
        if similarity_matrix[i][j] > threshold
    ]
}

# Output the JSON data to a file
with open('artist_similarity_graph.json', 'w') as json_file:
    json.dump(json_data, json_file)

print("JSON data for D3.js visualization saved as 'artist_similarity_graph.json'")