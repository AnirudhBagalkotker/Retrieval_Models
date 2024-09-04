from collections import defaultdict
import math
import os
import pickle
import subprocess


def get_document_vectors(documents, term_weights):
    document_vectors = defaultdict(lambda: defaultdict(float))
    for doc_id, doc in enumerate(documents):
        for term in doc:
            document_vectors[doc_id][term] = term_weights.get(term, 0)
    return document_vectors


def get_centroid_vector(document_vectors, doc_ids):
    centroid = defaultdict(float)
    num_docs = len(doc_ids)
    for doc_id in doc_ids:
        for term, weight in document_vectors[doc_id].items():
            centroid[term] += weight
    for term in centroid:
        centroid[term] /= num_docs
    return centroid


def rocchio_feedback(
    query_vector, document_vectors, top_k=10, alpha=1.0, beta=0.8, gamma=0.2
):
    sorted_doc_ids = sorted(
        document_vectors.keys(),
        key=lambda doc_id: sum(document_vectors[doc_id].values()),
        reverse=True,
    )
    relevant_doc_ids = sorted_doc_ids[:top_k]
    non_relevant_doc_ids = sorted_doc_ids[top_k:]

    centroid_relevant = get_centroid_vector(document_vectors, relevant_doc_ids)
    centroid_non_relevant = get_centroid_vector(document_vectors, non_relevant_doc_ids)

    modified_query = defaultdict(float)
    for term in set(query_vector) | set(centroid_relevant) | set(centroid_non_relevant):
        modified_query[term] = (
            alpha * query_vector.get(term, 0)
            + beta * centroid_relevant.get(term, 0)
            - gamma * centroid_non_relevant.get(term, 0)
        )

    return modified_query


def retrieve_ranked_list(query_vector, document_vectors):
    scores = defaultdict(float)
    for doc_id, doc_vector in document_vectors.items():
        score = sum(query_vector[term] * doc_vector[term] for term in query_vector)
        scores[doc_id] = score
    ranked_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_list


def load_documents(data_dir):
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


def load_queries(data_dir, query_type="titles"):
    query_file = f"{query_type}.queries"
    query_path = os.path.join(data_dir, query_file)
    queries = {}
    with open(query_path, "r", encoding="utf-8") as file:
        for line in file:
            query_id, query_text = line.strip().split("\t")
            queries[query_id] = query_text
    return queries


# main fn
def main():
    data_dir = "./nfcorpus/raw"
    documents = load_documents(data_dir)
    # query = input("Enter your query: ")
    # query = load_queries(data_dir, query_type="titles")
    query = "model"
    with open("term_weights.pkl", "rb") as file:
        term_weights = pickle.load(file)

    # Convert the query to a vector representation
    query_vector = defaultdict(float)
    for term in query.split():
        query_vector[term] += 1

    document_vectors = get_document_vectors(documents, term_weights)
    modified_query = rocchio_feedback(query_vector, document_vectors)

    print(f"Modified query: {modified_query}")

    # Retrieve a ranked list of documents for the query
    ranked_list = retrieve_ranked_list(query_vector, document_vectors)

    # Print the ranked list (document IDs and scores)
    for rank, (doc_id, score) in enumerate(ranked_list, start=1):
        print(f"Rank {rank}: Document ID {doc_id}, Score {score}")

    # Write the ranked list to a file for evaluation
    with open("ranked_list_rocchio.txt", "w") as file:
        for doc_id, score in ranked_list:
            file.write(f"{query} Q0 {doc_id} 0 {score} my_system\n")

    # Run trec_eval to evaluate the ranked list
    # subprocess.run(["trec_eval", "-q", "nfcorpus/merged.qrel", "ranked_list_rocchio.txt"])


if __name__ == "__main__":
    main()
