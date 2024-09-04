import re
import math
import os
import pickle
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# # Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_data(documents):
    """
    Preprocesses the given documents by tokenizing them, converting them to lowercase, removing punctuation,
    removing stopwords, and stemming the tokens.

    Parameters:
    - documents (List[str]): A list of documents to be preprocessed.

    Returns:
    - stemmed_docs (List[List[str]]): A list of lists, where each inner list contains the stemmed tokens
        for each token of the corresponding document.
    """
    # Tokenization
    tokenized_docs = [doc.split() for doc in documents]

    # Convert to lowercase
    lowercase_docs = [[token.lower() for token in doc] for doc in tokenized_docs]

    # Remove punctuation
    punctuation_pattern = r"[^\w\s]"
    cleaned_docs = []
    for doc in lowercase_docs:
        cleaned_doc = []
        for token in doc:
            cleaned_token = re.sub(punctuation_pattern, "", token)
            if cleaned_token:
                cleaned_doc.append(cleaned_token)
        cleaned_docs.append(cleaned_doc)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_docs = [
        [token for token in doc if token not in stop_words] for doc in cleaned_docs
    ]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_docs = [[stemmer.stem(token) for token in doc] for doc in filtered_docs]

    return stemmed_docs


def calculate_term_frequencies(processed_docs):
    """
    Calculate the term frequencies in a collection of processed documents.

    Parameters:
    - processed_docs (list): A list of lists, where each inner list represents a document and contains the processed terms.

    Returns:
    - term_frequencies (defaultdict): A nested defaultdict where the keys are terms and the values are defaultdicts where the keys are document IDs and the values are the frequencies of the terms in each document.

    Example:
    ```
    processed_docs = [
        ["apple", "banana", "orange"],
        ["banana", "kiwi", "mango"],
        ["apple", "kiwi", "orange"]
    ]
    term_frequencies = calculate_term_frequencies(processed_docs)
    print(term_frequencies)
    # defaultdict(<function __main__.lambda at 0x7ff911f20960>, {
    #     'apple': defaultdict(<class 'int'>, {0: 1, 2: 1}),
    #     'banana': defaultdict(<class 'int'>, {0: 1, 1: 1}),
    #     'orange': defaultdict(<class 'int'>, {0: 1, 2: 1}),
    #     'kiwi': defaultdict(<class 'int'>, {1: 1, 2: 1}),
    #     'mango': defaultdict(<class 'int'>, {1: 1,})
    # })
    ```
    """
    term_frequencies = defaultdict(lambda: defaultdict(int))
    for doc_id, doc in enumerate(processed_docs):
        for term in doc:
            term_frequencies[term][doc_id] += 1
    return term_frequencies


def calculate_document_frequencies(term_frequencies):
    """
    Calculate the document frequencies based on the term frequencies.

    Parameters:
    - term_frequencies (dict): A dictionary where the keys are terms and the values are dictionaries representing the frequency of each term in each document.

    Returns:
    - document_frequencies (defaultdict): A defaultdict where the keys are terms and the values are the number of documents in which the term appears.
    """
    document_frequencies = defaultdict(int)
    for term, doc_freqs in term_frequencies.items():
        document_frequencies[term] = len(doc_freqs)
    return document_frequencies


def calculate_collection_frequencies(term_frequencies):
    """
    Calculate the collection frequencies based on the term frequencies.

    Parameters:
    - term_frequencies (dict): A dictionary where the keys are terms and the values are dictionaries representing the frequency of each term in each document.

    Returns:
    - collection_frequencies (defaultdict): A defaultdict where the keys are terms and the values are the total frequencies of the terms across all documents.
    """
    collection_frequencies = defaultdict(int)
    for term, doc_freqs in term_frequencies.items():
        collection_frequencies[term] = sum(doc_freqs.values())
    return collection_frequencies


def represent_data(term_frequencies, document_frequencies, collection_frequencies):
    """
    Generates representations of term frequencies, document frequencies, and collection frequencies.

    Args:
        term_frequencies (dict): A dictionary where the keys are terms and the values are dictionaries representing the frequency of each term in each document.
        document_frequencies (dict): A dictionary where the keys are terms and the values are the number of documents in which the term appears.
        collection_frequencies (dict): A dictionary where the keys are terms and the values are the total frequencies of the terms across all documents.

    Returns:
        tuple: A tuple containing three lists representing the representations of term frequencies, document frequencies, and collection frequencies.
            - nnn_representation (list): A list of tuples containing the term frequency, document frequency, and collection frequency for each term in each document.
            - ntn_representation (list): A list of tuples containing the term frequency, term, and document frequency for each term in each document.
            - ntc_representation (list): A list of tuples containing the term frequency, term, and collection frequency for each term in each document.
    """
    nnn_representation = []
    ntn_representation = []
    ntc_representation = []

    for term, doc_freqs in term_frequencies.items():
        df = document_frequencies[term]
        cf = collection_frequencies[term]
        for doc_id, tf in doc_freqs.items():
            nnn_representation.append((tf, df, cf))
            ntn_representation.append((tf, term, df))
            ntc_representation.append((tf, term, cf))

    return nnn_representation, ntn_representation, ntc_representation


def calculate_term_weights(nnn_representation, ntc_representation):
    """
    Calculates the weights of terms based on the term frequency, document frequency, and collection frequency.

    Args:
        nnn_representation (list): A list of tuples containing the term frequency, document frequency, and collection frequency for each term in each document.
        ntc_representation (list): A list of tuples containing the term frequency, term, and collection frequency for each term in each document.

    Returns:
        dict: A dictionary where the keys are terms and the values are the calculated term weights.
    """
    term_weights = {}
    max_term_freq = max(nnn_representation, key=lambda x: x[0])[0]
    for tf, df, cf in nnn_representation:
        term = next(
            (token for token, _, coll_freq in ntc_representation if coll_freq == cf),
            None,
        )
        if term is not None:
            tfidf = (tf / max_term_freq) * math.log(len(nnn_representation) / df)
            term_weights[term] = tfidf
    return term_weights


def calculate_document_similarities(document_vectors):
    """
    Calculate the similarity between pairs of documents using the cosine similarity metric.

    Args:
        document_vectors (dict): A dictionary where the keys are document names and the values are dictionaries representing the term frequencies for each document.

    Returns:
        dict: A dictionary where the keys are tuples of document names and the values are the cosine similarity scores between the corresponding documents.
    """
    document_similarities = {}
    for doc1, doc1_vector in document_vectors.items():
        for doc2, doc2_vector in document_vectors.items():
            if doc1 != doc2:
                dot_product = sum(
                    doc1_vector[term] * doc2_vector[term]
                    for term in set(doc1_vector) & set(doc2_vector)
                )
                doc1_norm = math.sqrt(sum(value**2 for value in doc1_vector.values()))
                doc2_norm = math.sqrt(sum(value**2 for value in doc2_vector.values()))
                similarity = dot_product / (doc1_norm * doc2_norm)
                document_similarities[(doc1, doc2)] = similarity
    return document_similarities


def load_documents(data_dir):
    """
    Load documents from a directory.

    This function reads a text file named "doc_dump.txt" from the specified directory and parses its contents. Each line in the file represents a document and is expected to be tab-separated. The function extracts the document ID, URL, title, and abstract from each line and concatenates the title and abstract into a single string, which is then added to the list of documents.

    Args:
        data_dir (str): The path to the directory containing the "doc_dump.txt" file.

    Returns:
        list: A list of document texts. Each element in the list represents a document and is a string containing the title and abstract concatenated.

    """
    documents = []
    for filename in ["doc_dump.txt"]:
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                doc_parts = line.strip().split("\t")
                if len(doc_parts) >= 4:
                    doc_id, doc_url, doc_title, doc_abstract = doc_parts[:4]
                    doc_text = "\n".join([doc_title, doc_abstract])
                    documents.append(doc_text)
    return documents


def main():
    """
    Driver code for processing documents, calculating term frequencies, document frequencies, and collection frequencies, representing data, calculating term weights, and saving the term weights to a file.
    """
    data_dir = "./nfcorpus/raw"
    documents = load_documents(data_dir)
    processed_docs = preprocess_data(documents)
    term_frequencies = calculate_term_frequencies(processed_docs)
    document_frequencies = calculate_document_frequencies(term_frequencies)
    collection_frequencies = calculate_collection_frequencies(term_frequencies)
    nnn, ntn, ntc = represent_data(
        term_frequencies, document_frequencies, collection_frequencies
    )

    # print(nnn)
    # print(ntn)
    # print(ntc)

    term_weights = calculate_term_weights(nnn, ntc)
    print("Term weights calculated.")

    with open("term_weights.pkl", "wb") as file:
        pickle.dump(term_weights, file)
    print("Term weights saved to term_weights.pkl.")


if __name__ == "__main__":
    main()
