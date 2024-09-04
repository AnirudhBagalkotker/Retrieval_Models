import shutil
import os
import json
import numpy as np
import cProfile
from memory_profiler import profile as mem_profile
import subprocess
import time
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# # Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# reads s2 corpus in json and
# creates an intermediary file
# containing token and doc_id pairs.


def read_json_corpus(json_path):
    """
    Reads a JSON corpus from the specified json_path, and writes the extracted data to an output.tsv file in the intermediate directory within the json_path.
    """
    f = open(json_path + "/s2_doc.json", encoding="utf-8")
    json_file = json.load(f)
    if not os.path.exists(json_path + "/intermediate/"):
        os.mkdir(json_path + "/intermediate/")
    o = open(json_path + "/intermediate/output.tsv", "w", encoding="utf-8")
    for json_object in json_file["all_papers"]:
        doc_no = json_object["docno"]
        title = json_object["title"][0]
        paper_abstract = json_object["paperAbstract"][0]
        tokens = title.split(" ")
        for t in tokens:
            o.write(t.lower() + "\t" + str(doc_no) + "\n")
        tokens = paper_abstract.split(" ")
        for t in tokens:
            o.write(t.lower() + "\t" + str(doc_no) + "\n")
    o.close()


# sorts (token, doc_id) pairs
# by token first and then doc_id
def sort(dir):
    f = open(dir + "/intermediate/output.tsv", encoding="utf-8")
    o = open(dir + "/intermediate/output_sorted.tsv", "w", encoding="utf-8")

    # initialize an empty list of pairs of
    # tokens and their doc_ids
    pairs = []

    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        if len(split_line) == 2:
            pair = (split_line[0], split_line[1])
            pairs.append(pair)

    # sort (token, doc_id) pairs by token first and then doc_id
    sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))

    # write sorted pairs to file
    for sp in sorted_pairs:
        o.write(sp[0] + "\t" + sp[1] + "\n")
    o.close()


# converts (token, doc_id) pairs
# into a dictionary of tokens
# and an adjacency list of doc_id
def construct_postings(dir):
    # open file to write postings
    o1 = open(dir + "/intermediate/postings.tsv", "w", encoding="utf-8")

    postings = {}  # initialize our dictionary of terms
    doc_freq = {}  # document frequency for each term

    # read the file containing the sorted pairs
    f = open(dir + "/intermediate/output_sorted.tsv", encoding="utf-8")

    # initialize sorted pairs
    sorted_pairs = []

    # read sorted pairs
    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        pairs = (split_line[0], split_line[1])
        sorted_pairs.append(pairs)

    # construct postings from sorted pairs
    for pairs in sorted_pairs:
        if pairs[0] not in postings:
            postings[pairs[0]] = []
            postings[pairs[0]].append(pairs[1])
        else:
            len_postings = len(postings[pairs[0]])
            if len_postings >= 1:
                # check for duplicates
                # assuming the doc_ids are sorted
                # the same doc_ids will appear
                # one after another and detected by
                # checking the last element of the postings
                if pairs[1] != postings[pairs[0]][len_postings - 1]:
                    postings[pairs[0]].append(pairs[1])

    # update doc_freq which is the size of postings list
    for token in postings:
        doc_freq[token] = len(postings[token])

    # print("postings: " + str(postings))
    # print("doc freq: " + str(doc_freq))
    print("Dictionary size: " + str(len(postings)))

    # write postings and document frequency to file

    for token in postings:
        o1.write(token + "\t" + str(doc_freq[token]))
        for l in postings[token]:
            o1.write("\t" + l)
        o1.write("\n")
    o1.close()


def construct_trie(dir):
    # Initialize an empty trie
    trie = defaultdict(dict)

    # Read the file containing token and doc_id pairs
    with open(dir + "/intermediate/output_sorted.tsv", "r", encoding="utf-8") as file:
        for line in file:
            # Split the line into token and doc_id
            parts = line.strip().split("\t")

            # Check if the line has enough parts
            if len(parts) >= 2:
                token, doc_id = parts

                # Insert the token into the trie
                current_node = trie
                for char in token:
                    current_node = current_node.setdefault(char, {})

                # Add the doc_id to the leaf node
                current_node.setdefault("documents", []).append(doc_id)

    # Save the trie to a JSON file
    with open(dir + "/intermediate/trie.json", "w", encoding="utf-8") as json_file:
        # Convert sets to lists before serialization
        json.dump(
            trie, json_file, default=lambda o: list(o) if isinstance(o, set) else o
        )


# starting the indexing process
def index(dir):
    # reads the corpus and
    # creates an intermediary file
    # containing token and doc_id pairs.
    # read_corpus(dir)
    read_json_corpus(dir)

    # sorts (token, doc_id) pairs
    # by token first and then doc_id
    sort(dir)

    # converts (token, doc_id) pairs
    # into a dictionary of tokens
    # and an adjacency list of doc_id
    construct_postings(dir)
    construct_trie(dir)


def load_trie_in_memory(dir):
    f = open(dir + "/intermediate/trie.json", encoding="utf-8")
    trie = json.load(f)
    doc_freq = {}

    return trie, doc_freq


def load_index_in_memory(dir):
    """
    Load index data from the specified directory into memory.

    Parameters:
    dir (str): The directory path where the index data is located.

    Returns:
    tuple: A tuple containing two dictionaries - postings and doc_freq.
            postings (dict): A dictionary containing token as key and list of items as value.
            doc_freq (dict): A dictionary containing token as key and frequency as value.
    """
    f = open(dir + "intermediate/postings.tsv", encoding="utf-8")
    postings = {}
    doc_freq = {}

    for line in f:
        splitline = line.split("\t")

        token = splitline[0]
        freq = int(splitline[1])

        doc_freq[token] = freq

        item_list = []

        for item in range(2, len(splitline)):
            item_list.append(splitline[item].strip())
        postings[token] = item_list

    return postings, doc_freq


def intersection(l1, l2):
    """
    Calculate the intersection of two lists.

    Args:
        l1: The first list.
        l2: The second list.

    Returns:
        A new list containing the intersection of l1 and l2.
    """
    count1 = 0
    count2 = 0
    intersection_list = []

    while count1 < len(l1) and count2 < len(l2):
        if l1[count1] == l2[count2]:
            intersection_list.append(l1[count1])
            count1 = count1 + 1
            count2 = count2 + 1
        elif l1[count1] < l2[count2]:
            count1 = count1 + 1
        elif l1[count1] > l2[count2]:
            count2 = count2 + 1

    return intersection_list


def and_query(query_terms, corpus):
    """
    A function to perform an 'and' query on a given set of query terms against a corpus.
    Parameters:
        query_terms (list): List of query terms to search for in the corpus
        corpus (str): The text corpus to search within
    Returns:
        list: A list of document IDs that contain all the query terms
    """
    # load postings in memory
    postings, doc_freq = load_index_in_memory(corpus)

    # postings for only the query terms
    postings_for_keywords = {}
    doc_freq_for_keywords = {}

    for q in query_terms:
        # Check if the query term is in the index
        if q in postings:
            postings_for_keywords[q] = postings[q]
            doc_freq_for_keywords[q] = doc_freq[q]
        else:
            # If the query term is not in the index, set empty postings
            postings_for_keywords[q] = []
            doc_freq_for_keywords[q] = 0

    # for q in query_terms:
    #     postings_for_keywords[q] = postings[q]

    # # store doc frequency for query token in
    # # dictionary

    # for q in query_terms:
    #     doc_freq_for_keywords[q] = doc_freq[q]

    # sort tokens in increasing order of their
    # frequencies

    sorted_tokens = sorted(doc_freq_for_keywords.items(), key=lambda x: x[1])

    # initialize result to postings list of the
    # token with minimum doc frequency

    result = postings_for_keywords[sorted_tokens[0][0]]

    # iterate over the remaining postings list and
    # intersect them with result, and updating it
    # in every step

    for i in range(1, len(postings_for_keywords)):
        result = intersection(result, postings_for_keywords[sorted_tokens[i][0]])
        if len(result) == 0:
            return result

    return result


def trie_search(trie, query):
    """
    Function to search for a query in a trie data structure.

    Parameters:
    - trie: the trie data structure to search in
    - query: the query string to search for in the trie

    Returns:
    - A set of documents at the leaf node matching the query
    """
    current_node = trie
    for char in query:
        if char not in current_node:
            return set()  # No matching documents for the query
        current_node = current_node[char]

    # Return the set of documents at the leaf node
    return set(current_node.get("documents", []))


def trie_and_query(trie, query_terms):
    """
    Function to perform a query using a trie data structure.

    Parameters:
    - trie: The trie data structure to be queried.
    - query_terms: The list of terms to be queried.

    Returns:
    - A set containing the results of the query.
    """
    result = set()
    count = 0
    for term in query_terms:
        if count == 0:
            result = trie_search(trie, term)
            count += 1
        else:
            result = result & trie_search(trie, term)

    return result


def run_queries_using_trie_and_query(queries_file, corpus_dir):
    # Load trie from JSON file
    with open(
        corpus_dir + "/intermediate/trie.json", "r", encoding="utf-8"
    ) as json_file:
        trie = json.load(json_file)

    queries = read_queries_from_file(queries_file)

    for query_info in queries:
        query_id = query_info["qid"]
        query_text = query_info["query"]

        query_terms = query_text.lower().split(" ")

        # Perform trie-based search and boolean retrieval
        result = trie_and_query(trie, query_terms)

        print(f"Query ID: {query_id}, Query: {query_text}, Result: {result}")


def read_queries_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        queries_data = json.load(file)
    return queries_data["queries"]


def build_trie(dir):
    """
    Builds a trie from the data in the specified directory.

    Parameters:
        dir (str): The directory containing the data.

    Returns:
        TrieNode: The root of the trie built from the data.
    """
    trie_root = TrieNode()

    f = open(dir + "intermediate/output.tsv", encoding="utf-8")
    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        if len(split_line) == 2:
            token, doc_id = split_line
            insert_to_trie(trie_root, token, doc_id)

    return trie_root


def insert_to_trie(root, token, doc_id):
    node = root
    for char in token:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.postings.append(doc_id)


def and_query_trie(query_terms, trie_root):
    result = set()

    for term in query_terms:
        result = set(traverse_trie(trie_root, term, result))

    return result


def traverse_trie(node, prefix, result):
    for char in prefix:
        if char in node.children:
            node = node.children[char]
        else:
            return result
    result.update(node.postings)
    return result


class TrieNode:
    def __init__(self):
        self.children = {}
        self.documents = []


def insert_to_trie(root, token, doc_id):
    node = root
    for char in token:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.documents.append(doc_id)


def construct_forward_trie(dictionary):
    forward_trie = TrieNode()
    for term in dictionary:
        current_node = forward_trie
        for char in term:
            current_node = current_node.children.setdefault(char, TrieNode())
            current_node.documents.add(term)
    return forward_trie


def construct_backward_trie(dictionary):
    backward_trie = TrieNode()
    for term in dictionary:
        current_node = backward_trie
        for char in reversed(term):
            current_node = current_node.children.setdefault(char, TrieNode())
            current_node.documents.add(term)
    return backward_trie


def forward_trie_search(trie, query):
    current_node = trie
    for char in query:
        if char not in current_node.children:
            return set()  # No matching documents for the query
        current_node = current_node.children[char]

    # Return the set of documents at the leaf node
    return set(current_node.documents)


def backward_trie_search(trie, query):
    current_node = trie
    for char in query:
        if char not in current_node.children:
            return set()  # No matching documents for the query
        current_node = current_node.children[char]

    # Return the set of documents at the leaf node
    return set(current_node.documents)


def load_data_into_trie_forward(trie_root, tsv_file_path):
    with open(tsv_file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                token, doc_id = parts
                insert_to_trie(trie_root, token, doc_id)


def load_data_into_trie_backward(trie_root, tsv_file_path):
    with open(tsv_file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                token, doc_id = parts
                insert_to_trie(trie_root, token[::-1], doc_id)


def construct_permuterm_index_from_trie(trie_root):
    permuterm_index = {}
    construct_permuterm_index_recursive(trie_root, "", permuterm_index)
    return permuterm_index


def construct_permuterm_index_recursive(node, current_term, permuterm_index):
    if not node.children:
        # If the node has no children, add the documents associated with the current node
        permuterm_index[current_term] = node.documents
    else:
        # Traverse the trie recursively and concatenate characters to build the permuterm representation
        for char, child_node in node.children.items():
            construct_permuterm_index_recursive(
                child_node, current_term + char, permuterm_index
            )


def wildcard_search_trie(trie, wildcard_query):
    """
    Performs a wildcard search in a trie data structure.

    Args:
        trie (Trie): The trie data structure to search in.
        wildcard_query (str): The wildcard query string to search for.

    Returns:
        set: A set of results matching the wildcard query.
    """
    current_node = trie
    results = set()

    def backtrack(node, current_result, wildcard_remaining):
        """
        Backtracking algorithm to process wildcard search on a trie data structure.

        Args:
            node: The current node in the trie.
            current_result: The current result string formed during backtracking.
            wildcard_remaining: The remaining wildcard string to be processed.

        Returns:
            None
        """
        if not wildcard_remaining:
            results.update(node.documents)
            return

        if not node.children:
            results.update(node.documents)
            return

        char = wildcard_remaining[0]

        if char == "*":
            for child_char in node.children:
                backtrack(node.children[child_char], current_result, wildcard_remaining)
        elif char in node.children:
            backtrack(
                node.children[char], current_result + char, wildcard_remaining[1:]
            )

    backtrack(current_node, "", wildcard_query)
    return results


def write_permuterm_index_to_file(permuterm_index, output_file):
    """
    Write the permuterm index to a file.

    Args:
        permuterm_index: The permuterm index to write to the file.
        output_file: The file to write the permuterm index to.

    Returns:
        None
    """
    with open(output_file, "w", encoding="utf-8") as file:
        for term, documents in permuterm_index.items():
            file.write(f"{term} {documents}\n")


def read_queries_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        queries_data = json.load(file)
    return queries_data["queries"]


def boolean_from_trie():
    """
    A function to load a trie from a JSON file, read queries from another file,
    and process each query to obtain results from the trie.
    """
    with open("s2/intermediate/trie.json", "r", encoding="utf-8") as json_file:
        trie = json.load(json_file)

    queries = read_queries_from_file("s2/s2_query.json")

    for query_info in queries:
        query_id = query_info["qid"]
        query_text = query_info["query"]

        query_terms = query_text.lower().split(" ")
        result = trie_and_query(trie, query_terms)
        # print(f"Query: {query_text}, Results: {result}")
        # document_reading(query_text, query_id, result)


def construct_permuterm_index_from_trie(trie):
    """
    Constructs a permuterm index from the given trie.

    Parameters:
        trie: The trie data structure to construct the permuterm index from.

    Returns:
        dict: The permuterm index containing the permutations of terms and corresponding documents.
    """
    permuterm_index = {}

    def generate_permutations(node, current_term):
        """
        Generate permutations for a given node and current term.

        Args:
            node: The current node in the trie.
            current_term: The current term being constructed.

        Returns:
            None
        """
        """
        Generate permutations for a given node and current term.

        Args:
            node: The current node in the trie.
            current_term: The current term being constructed.

        Returns:
            None
        """
        if node.documents:
            # Add the current term to the permuterm index
            permuterm_index[current_term] = node.documents

        for char, child_node in node.children.items():
            generate_permutations(child_node, current_term + char)

    generate_permutations(trie, "")

    return permuterm_index


def main_permute_index():
    """
    This function reads a JSON file containing a trie, constructs a permuterm index from the trie, and writes the permuterm index to a TSV file.
    """
    with open("s2/intermediate/trie.json", "r", encoding="utf-8") as json_file:
        trie = json.load(json_file)
    permute_index = construct_permuterm_index_from_trie(forward_trie)
    write_permuterm_index_to_file(permute_index, "s2/intermediate/permute_index.tsv")


def main_wildcard():
    """
    This function reads wildcard queries from a file, performs wildcard search, and processes the results.
    """

    queries = read_queries_from_file("s2/s2_wildcard.json")
    wildcard_queries = []
    for query_info in queries:
        query_id = query_info["qid"]
        query_text = query_info["query"]
        wildcard_queries.append(query_text)

    for query in wildcard_queries:
        forward_result = wildcard_search_trie(forward_trie, query)
        backward_result = wildcard_search_trie(backward_trie, query[::-1])
        result = forward_result & backward_result
        # print(f"Wildcard: {query}, Results: {result}")
        # document_reading(query_text, query_id, result)


def read_queries_from_file(file_path):
    """
    Reads queries from a file and returns the queries data.

    Parameters:
        file_path (str): The path to the file containing the queries.

    Returns:
        list: The list of queries.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        queries_data = json.load(file)
    return queries_data["queries"]


def tolerant_retrieval():
    """
    Function for tolerant retrieval. Reads queries from a file, processes the queries, and performs wildcard search and and_query operations to retrieve results. Does not return any value.
    """

    queries = read_queries_from_file("s2/s2_wildcard_boolean.json")

    for query_info in queries:
        query_id = query_info["qid"]
        query_text = query_info["query"]

        query_terms = query_text.lower().split(" ")
        non_wild_list = []
        wild_list = []
        for q in query_terms:
            val = q.find("*")
            if val == -1:
                non_wild_list.append(q)
            else:
                wild_list.append(q)
        # print(non_wild_list)
        resultfinal = []
        if len(wild_list) != 0 and len(non_wild_list) != 0:
            for qu in wild_list:
                forward_result = wildcard_search_trie(forward_trie, qu)
                backward_result = wildcard_search_trie(backward_trie, qu[::-1])
                result2 = list(forward_result & backward_result)
                result2.sort()
            result1 = and_query(non_wild_list, "s2/")
            resultfinal = intersection(result2, result1)
        elif len(wild_list) == 0 and len(non_wild_list) != 0:
            resultfinal = and_query(non_wild_list, "s2/")
        elif len(wild_list) != 0 and len(non_wild_list) == 0:
            for qu in wild_list:
                forward_result = wildcard_search_trie(forward_trie, qu)
                backward_result = wildcard_search_trie(backward_trie, qu[::-1])
                result2 = list(forward_result & backward_result)
                result2.sort()
                resultfinal = result2
        print(f"Wildcard Query: {query_text}, Results: {resultfinal}")
        # document_reading(query_text, query_id, resultfinal)


def run_grep(query, corpus_dir):
    """
    Perform a recursive grep search for the given query in the specified corpus directory.

    Parameters:
        query (str): The string to search for.
        corpus_dir (str): The directory in which to search.

    Returns:
        None
    """
    command = f'grep -r "{query}" {corpus_dir}'
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)

    print(result.stdout)


def grep_run_queries(queries_file_path, corpus_dir):
    """
    Execute a series of queries from the specified file on the provided corpus directory using the run_grep function.

    Parameters:
        queries_file_path (str): The file path to the queries file.
        corpus_dir (str): The directory containing the corpus data.

    Returns:
        None
    """
    with open(queries_file_path, "r") as file:
        queries_data = json.load(file)

    for query_info in queries_data["queries"]:
        qid = query_info["qid"]
        query = query_info["query"]

        # print(f"Results for query {qid} = '{query}': ")
        run_grep(query, corpus_dir)
        # print("\n")


def run_queries_from_file(queries_file, corpus_dir):
    """
    Runs queries from a file and processes them against the provided corpus directory.

    Args:
        queries_file: The file containing the queries to be processed.
        corpus_dir: The directory containing the corpus to be queried.

    Returns:
        None
    """
    postings, doc_freq = load_index_in_memory(corpus_dir)

    queries = read_queries_from_file(queries_file)

    for query_info in queries:
        query_id = query_info["qid"]
        query_text = query_info["query"]

        query_terms = query_text.lower().split(" ")

        result = and_query(query_terms, corpus_dir)
        # document_reading(query_text, query_id, result)
        # if(query_id == "1"):
        #     print(f"Forward Wildcard Query: {query_terms}, Results: {result}")
        # print(f"Query ID: {query_id}, Query: {query_text}, Result: {result}")


def profiled_code_grep():
    """
    This function performs a profiled code grep. It takes no parameters and does not return anything.
    """

    queries_file_path = "s2/s2_query.json"

    grep_run_queries(queries_file_path, "s2")


def profiled_code():
    """
    This function runs a set of queries from a specified file path and profiles the code.
    """

    queries_file_path = "s2/s2_query.json"

    run_queries_from_file(queries_file_path, "s2/")


build_trie("s2/")
forward_trie = TrieNode()
backward_trie = TrieNode()

# Insert data into the trie from output_sorted.tsv
load_data_into_trie_forward(forward_trie, "s2/intermediate/output_sorted.tsv")
load_data_into_trie_backward(backward_trie, "s2/intermediate/output_sorted.tsv")


def read_docs(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        doc_data = json.load(file)
    return doc_data["all_papers"]


docs = read_docs("s2/s2_doc.json")


def document_reading(query_text, qid, list):
    print(f"Query ID: {qid}, Query: {query_text}")
    for doc_info in docs:
        docNo = doc_info["docno"]
        title = doc_info["title"]
        paper_abstract = doc_info["paperAbstract"]

        if docNo in list:
            if len(paper_abstract) != 0:
                print(paper_abstract)
            else:
                print(title)


def linguistic_post_processing(vocabulary):
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Get English stopwords
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(vocabulary)

    processed_vocabulary = []

    for word in tokens:
        # Stemming
        stemmed_word = stemmer.stem(word)

        # Lemmatization
        lemmatized_word = lemmatizer.lemmatize(word)

        # Stop word removal
        if lemmatized_word.lower() not in stop_words:
            processed_vocabulary.append(lemmatized_word.lower())

    return processed_vocabulary


if __name__ == "__main__":
    index("s2/")
    cProfile.run("profiled_code()", sort="cumulative")
    cProfile.run("profiled_code_grep()", sort="cumulative")
    cProfile.run("boolean_from_trie()", sort="cumulative")
    cProfile.run("main_permute_index()", sort="cumulative")
    cProfile.run("main_wildcard()", sort="cumulative")
    cProfile.run("tolerant_retrieval()", sort="cumulative")
