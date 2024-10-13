from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
import matplotlib
import matplotlib.pyplot as plt 

matplotlib.use('Agg')

# download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# load dataset
newsgroups = fetch_20newsgroups(subset='all')

def preprocess_text(text):
    return ' '.join([word.lower() for word in text.split() if word.lower() not in stop_words])

def apply_lsa_news(n_components=100):
    '''
    Returns the vectorizer, SVD model, and SVD-reduced matrix for the newsgroup dataset
    '''
    # pre-process text
    news_data = [preprocess_text(news) for news in newsgroups['data']]
    
    # get term-document matrix
    vectorizer = TfidfVectorizer(max_df=0.95, max_features=1000)
    term_doc_matrix = vectorizer.fit_transform(news_data)

    # apply SVD
    svd_model = TruncatedSVD(n_components=n_components)
    svd_matrix = svd_model.fit_transform(term_doc_matrix)

    return vectorizer, svd_model, svd_matrix

# initialize LSA components
vectorizer, svd_model, svd_matrix = apply_lsa_news()

def process_search(input_text):
    '''
    Returns cosine similarity of input query and newsgroup data
    '''
    # process input as a vector 
    input_vec = vectorizer.transform([input_text])
    
    # transform input vector to the reduced SVD space
    input_vec_reduced = svd_model.transform(input_vec)

    # get cosine similarities between the query and the SVD matrix
    similarities = cosine_similarity(input_vec_reduced, svd_matrix)

    return similarities

def get_top_documents(search, n=6):
    '''
    Retrieves the top 'n' documents based on similarity to the search query
    '''
    # Get cosine similarities
    similarities = process_search(search)

    # List indices of the top matches
    top_indices = similarities.argsort()[0][-n:][::-1]
    top_docs = [(newsgroups.data[i], similarities[0][i]) for i in top_indices]
    sim_scores = [similarities[0][i] for i in top_indices]
    
    return top_docs, sim_scores, top_indices

def get_graph(sim_scores, top_indices):
    fig = plt.figure(figsize = (15, 8))
    x_labels = [("Document " + str(i)) for i in top_indices]
    
    plt.bar(x_labels, sim_scores)

    plt.title("Document and Cosine Similarity Scores")
    plt.xlabel("Document")
    plt.ylabel("Cosine Similarity Score")

    plt.savefig('static/result.png')
    plt.close()

# # testing
# top_docs, sim_scores, top_indices = get_top_documents("machine learning", n=5)
# for i in range(len(top_docs)):
#     print(f"indices: {top_indices[i]}\ncosine similarity score: {sim_scores[i]}\ndocument text: {top_docs[i][0][:50]}...\n")
