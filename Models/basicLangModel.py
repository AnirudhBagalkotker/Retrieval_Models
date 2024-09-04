from collections import defaultdict
import os


class LanguageModel:
    def __init__(self, documents):
        self.term_frequencies = defaultdict(int)
        self.document_lengths = defaultdict(int)
        self.total_terms = 0

        self.build_index(documents)

    def build_index(self, documents):
        for doc_id, document in enumerate(documents):
            terms = document.split()  # Tokenize the document
            for term in terms:
                self.term_frequencies[term] += 1
                self.document_lengths[doc_id] += 1
                self.total_terms += 1

    def score(self, query, document_id, mu=500):
        score = 0
        for term in query:
            if term in self.term_frequencies:
                score += (
                    self.term_frequencies[term]
                    + mu * self.term_probability(term, document_id)
                ) / (self.document_lengths[document_id] + mu)
        return score

    def term_probability(self, term, document_id):
        return self.term_frequencies[term] / self.total_terms


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
    lm = LanguageModel(documents)
    scores = {doc_id: lm.score(query, doc_id) for doc_id in range(len(documents))}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    ranking = [doc_id for doc_id, score in sorted_scores]

    print("Ranking of documents:")
    for rank, doc_id in enumerate(ranking, start=1):
        print(f"Rank {rank}: Document {doc_id}, Score: {scores[doc_id]}")

    # Write the ranked list to a file for evaluation
    with open("ranked_list_blm.txt", "w") as file:
        for doc_id, score in ranking:
            file.write(f"{query} Q0 {doc_id} 0 {score} my_system\n")

    # Run trec_eval to evaluate the ranked list
    # subprocess.run(["trec_eval", "-q", "nfcorpus/merged.qrel", "ranked_list_blm.txt"])


if __name__ == "__main__":
    main()
