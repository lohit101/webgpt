# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import random

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the URL of the website's about page
url = 'https://www.cdp.net/en/info/about-us'

# Send an HTTP GET request to the URL
try:
    response = requests.get(url)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Failed to retrieve the website content. Error: {e}")
    exit(1)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text content from the HTML
    text = soup.get_text()

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Define a list of stopwords
    stop_words = set(stopwords.words('english'))

    # Process and analyze the sentences
    for i, sentence in enumerate(sentences):
        # Tokenize words
        words = nltk.word_tokenize(sentence)

        # Remove stopwords
        words = [word.lower() for word in words if word.lower() not in stop_words]
        sentences[i] = ' '.join(words)

    # Use TF-IDF vectorization and LSA for analysis
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

    # Apply Latent Semantic Analysis (LSA)
    num_topics = 5  # You can adjust the number of topics
    lsa = TruncatedSVD(n_components=num_topics, n_iter=100)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    # Initialize a conversation loop
    conversation_history = []
    print("Bot: Hello! Ask me anything about this website's about page. Type 'quit' to exit.")

    while True:
        user_input = input("You: ")

        # Exit the loop if the user wants to quit
        if user_input.lower() == "quit":
            break

        # Process the user input
        processed_input = user_input.lower()

        # Calculate the TF-IDF vector for user input
        input_vector = tfidf_vectorizer.transform([processed_input])

        # Apply LSA to user input
        input_lsa = lsa.transform(input_vector)

        # Calculate cosine similarity between user input and website content in LSA space
        similarity_scores = cosine_similarity(input_lsa, lsa_matrix)

        # Sort and get the most relevant sentences
        most_relevant_indices = similarity_scores[0].argsort()[::-1]
        relevant_sentences = [sentences[i] for i in most_relevant_indices]

        # Generate a dynamic response
        response = ""

        if len(relevant_sentences) > 0:
            response = random.choice(relevant_sentences[:3])  # Display a random relevant sentence
        else:
            response = "I couldn't find any specific information related to your input."

        # Print the chatbot's response
        print("Bot:", response)

    # Thank the user and exit
    print("Bot: Thank you for the conversation. Goodbye!")

else:
    print("Failed to retrieve the website content.")