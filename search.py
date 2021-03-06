#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import json
import heapq
from nltk.tokenize import *
from math import log10

stemmer = nltk.stem.PorterStemmer()


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    print('running search on the queries...')

    with open(dict_file, 'r') as f:
        global_dict = json.load(f)  # Dictionary in the form 'term: [termid, doc_freq, byte_offset]'
    postings_fd = open(postings_file, 'r')  # open in read mode
    queries_fd = open(queries_file, 'r')  # open in read mode
    queries_list = queries_fd.read().splitlines()
    result_list = []
    for query in queries_list:
        tokenized_query = parse_query(query, global_dict)                   # tokenize and process query
        score = compute_score(tokenized_query, global_dict, postings_fd)    # add scores
        if not score:   # no valid docIDs found, so just append an empty list of docIDs to result
            result_list.append([])
            continue
        top_10_docs = get_top_docs(score)
        result_list.append(top_10_docs)

    # write results to output file
    i = 0
    result_len_minus_one = len(result_list) - 1
    with open(results_file, 'w') as r_file:
        for result in result_list:
            result_str = ''
            if result:  # there are valid docIDs found in result
                for posting in result:
                    result_str += str(posting) + ' '
            if i == result_len_minus_one:
                r_file.write(result_str.strip())    # do not add newline for last result
                continue
            r_file.write(result_str.strip() + "\n")
            i += 1
    r_file.close()

    return


def parse_query(query, global_dict):
    word_tokenized_query = word_tokenize(query)
    alnum_words = [word for word in word_tokenized_query if word.isalnum()]     # only keep alphanumeric terms
    stemmed_tokens = [stemmer.stem(word) for word in alnum_words]               # stem words
    stemmed_lower_tokens = [token.lower() for token in stemmed_tokens]          # convert tokens to lowercase

    # filter out terms in query that are not in dictionary
    filtered_tokens = []
    for token in stemmed_lower_tokens:
        if token in global_dict:
            filtered_tokens.append(token)
    return filtered_tokens


def compute_score(tokenized_query, global_dict, postings_fd):
    if not tokenized_query:     # no tokens available
        return []
    score = {}
    query_ltc_scores = compute_ltc_scores(tokenized_query, global_dict)
    for token in tokenized_query:
        # list of postings, with each posting being = [doc_id, tf-lnc]
        processed_postings_list = convert_term_to_postings(token, global_dict, postings_fd)
        query_score = query_ltc_scores[token]
        for posting in processed_postings_list:     # add lnc of each posting to score_dict
            doc_id = posting[0]
            if doc_id in score.keys():
                score[doc_id] += posting[1] * query_score
            else:
                score[doc_id] = posting[1] * query_score

    return score


def convert_term_to_postings(term, global_dict, postings_fd):
    """
    Retrieve the posting_list of the given term and convert it to an array of postings.
    """
    term_id, idf, skip_ptr = global_dict[term]
    postings_fd.seek(skip_ptr, 0)
    postings_string = postings_fd.readline()
    postings_string = postings_string.strip()      # strip any newline
    postings_split = postings_string.split()       # list of each posting with associated components

    # split each posting into [doc_id, tf-lnc]
    split_within_postings = []
    for posting in postings_split:
        posting_components = posting.split(',')
        # ignore posting_components[2] as it is just skip_ptr
        split_within_postings.append([int(posting_components[0]), float(posting_components[1])])
    return split_within_postings


def compute_ltc_scores(query_list, global_dict):
    term_ltc_scores = {}
    for query in query_list:
        if query in term_ltc_scores:
            term_ltc_scores[query] += 1
        else:
            term_ltc_scores[query] = 1

    for term, tf in term_ltc_scores.items():
        l = 1 + log10(tf)
        idf = global_dict[term][1]   # t
        term_ltc_scores[term] = l * idf

    # Get vector length
    sum_of_squares = 0
    for lt in term_ltc_scores.values():
        sum_of_squares += lt * lt
    vector_length = sum_of_squares ** 0.5

    # Normalize vector
    for term, lt in term_ltc_scores.items():
        term_ltc_scores[term] = lt / vector_length
    return term_ltc_scores


def get_top_docs(score):
    # convert key:value pairs in score to (-value, key) tuples.     Use -value as heapq in python is min heap
    score_docid_tuples = [(v * -1, k) for k, v in score.items()]

    # add all items in dictionary to a heap
    heap = []
    for score_docid in score_docid_tuples:
        heapq.heappush(heap, score_docid)

    # get top 10 docs
    result_list = []
    for i in range(10):
        if not heap:
            break
        score_docid = heapq.heappop(heap)
        result_list.append(score_docid[1])

    return result_list


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
