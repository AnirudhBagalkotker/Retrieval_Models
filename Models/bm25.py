from collections import defaultdict
import math
import os


class BM25:
    def __init__(self, documents):
        self.term_idf = {}
        self.document_lengths = []
        self.avg_document_length = 0
        self.total_documents = len(documents)

        self.calculate_idf(documents)
        self.calculate_avg_document_length(documents)

    def calculate_idf(self, documents):
        term_doc_freq = defaultdict(int)
        for document in documents:
            unique_terms = set(document.split())  # Split the document into terms
            for term in unique_terms:
                term_doc_freq[term] += 1

        for term, doc_freq in term_doc_freq.items():
            self.term_idf[term] = math.log(
                (self.total_documents - doc_freq + 0.5) / (doc_freq + 0.5)
            )

    def calculate_avg_document_length(self, documents):
        total_document_length = sum(len(doc) for doc in documents)
        self.avg_document_length = total_document_length / self.total_documents

    def score(self, query, document, k1=1.5, b=0.75):
        score = 0
        doc_length = len(document)
        for term in query:
            if term in document:
                tf = document.count(term)
                idf = self.term_idf.get(term, 0)
                score += (
                    idf
                    * (tf * (k1 + 1))
                    / (tf + k1 * (1 - b + b * doc_length / self.avg_document_length))
                )
        return score


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


def main():
    data_dir = "./nfcorpus/raw"
    documents = load_documents(data_dir)
    query = ["model", "risk"]
    bm25 = BM25(documents)
    scores = {
        doc_id: bm25.score(query, document) for doc_id, document in enumerate(documents)
    }
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    ranking = [doc_id for doc_id, score in sorted_scores]

    print("Ranking of documents:")
    for rank, doc_id in enumerate(ranking, start=1):
        print(f"Rank {rank}: Document {doc_id}, Score: {scores[doc_id]}")

    # Write the ranked list to a file for evaluation
    with open("ranked_list_bm25.txt", "w") as file:
        for doc_id, score in ranking:
            file.write(f"{query} Q0 {doc_id} 0 {score} my_system\n")

    # Run trec_eval to evaluate the ranked list
    # subprocess.run(["trec_eval", "-q", "nfcorpus/merged.qrel", "ranked_list_bm25.txt"])


if __name__ == "__main__":
    main()
