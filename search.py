#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import json
import heapq
from nltk.tokenize import *

stemmer = nltk.stem.PorterStemmer()
UNIVERSAL = '_universal'


def convert_term_to_postings(term, global_dict, postings_fd):
    """
    Retrieve the posting_list of the given term and convert it to an array of postings.
    """
    term_id, idf, skip_ptr = global_dict[term]
    postings_fd.seek(skip_ptr, 0)
    postings_string = postings_fd.readline()
    postings_string = postings_string.strip()      # strip any newline
    postings_split = postings_string.split()       # list of each posting with associated components

    # split each posting into [doc_id, tf-lnc] or [doc_id, tf-lnc, skip_offset]
    split_within_postings = []
    for posting in postings_split:
        posting_components = posting.split(',')
        # ignore posting_components[2] as it is just skip_ptr
        split_within_postings.append([int(posting_components[0]), float(posting_components[1])])
    # print('full process: ', split_within_postings)
    return split_within_postings


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
        tokenized_query = parse_query(query)                                # tokenize and process query
        score = compute_score(tokenized_query, global_dict, postings_fd)    # add scores
        top_10_docs = get_top_docs(score)
        result_list.append(top_10_docs)

    # write results to output file
    with open(results_file, 'w') as r_file:
        for result in result_list:
            result_str = ''
            for posting in result:
                result_str += str(posting) + ' '
            r_file.write(result_str.strip() + "\n")
    r_file.close()

    return


def parse_query(query):
    word_tokenized_query = word_tokenize(query)
    alnum_words = [word for word in word_tokenized_query if word.isalnum()]     # only keep alphanumeric terms
    stemmed_tokens = [stemmer.stem(word) for word in alnum_words]               # stem words
    stemmed_lower_tokens = [token.lower() for token in stemmed_tokens]          # convert tokens to lowercase
    return stemmed_lower_tokens


def compute_score(tokenized_query, global_dict, postings_fd):
    score = {}
    query_ltc_scores = compute_ltc_scores(tokenized_query, global_dict)
    i = 0
    for token in tokenized_query:
        # list of postings, with each posting being = [doc_id, tf-lnc]
        processed_postings_list = convert_term_to_postings(token, global_dict, postings_fd)
        # TODO do i have to compute w-t.d. * w-t.q. (using ltc)? W8 slides say we can assume w-t.q. to be 1. See line 89
        query_score = query_ltc_scores[i]
        for posting in processed_postings_list:     # add lnc of each posting to score_dict
            doc_id = posting[0]
            if doc_id in score.keys():
                score[doc_id] += posting[1] * query_score  # TODO can i just use *1 instead of *query_score?
            else:
                score[doc_id] = posting[1] * query_score
        i += 1

    return score


def compute_ltc_scores(query_list, global_dict):
    ltc_scores = []
    for query in query_list:
        # l = 1 + log(1) = 1
        idf = global_dict[query][1]   # t
        ltc_scores.append(idf)

    # Get vector length
    sum_of_squares = 0
    for lt in ltc_scores:
        sum_of_squares += lt * lt
    vector_length = sum_of_squares ** 0.5

    # Normalize vecotr
    i = 0
    for lt in ltc_scores:
        ltc_scores[i] = lt / vector_length
        i += 1
    return ltc_scores


def get_top_docs(score):
    # convert key:value pairs in score to (-value, key) tuples.     Use -value as heapq in python is min heap
    print(score)
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
