#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import json
from nltk.tokenize import *
from collections import deque
from math import floor

stemmer = nltk.stem.PorterStemmer()
operator = ['ANDNOT', 'NOT', 'AND', 'OR']
operator_precedence = {'ANDNOT': 4, 'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}
bracket = ['(', ')']
UNIVERSAL = '_universal'


class Node:
    ptr: int
    val: int
    next: int
    skip: int
    isLast: bool

    def __init__(self, ptr: int, val: int, next: int, skip: int, is_last: bool):
        """
        ptr - the true offset (in terms of number of bytes) from the start of the file to the location of this value
        val - the val read at this offset
        next - the ptr to the next val, used for incrementing ptr
        skip - the ptr to the skipped val, if skip pointer exists here (equals 0 if skip ptr does not exist)
        is_last - a flag to indicate if this node is the last of the posting list
        """
        self.ptr = ptr
        self.val = val
        self.next = next
        self.skip = skip
        self.is_last = is_last


def get_node(fd, ptr):
    """
    Abstraction for getting a node from a posting list

    Note:
    This function can be used to get nodes from a Python list as well.
    In which case, fd will be the list, ptr will be the idx
    """
    if isinstance(fd, list):
        n = len(fd)
        val = fd[ptr]
        is_last = ptr == n - 1
        next = ptr + 1 if ptr + 1 < n else ptr
        skip_dist = floor(n ** 0.5)
        skip = ptr + skip_dist if ptr + skip_dist < n else 0
        return Node(ptr, val, next, skip, is_last)

    fd.seek(ptr, 0)
    word = ''
    next_char = fd.read(1)
    while next_char != ' ' and next_char != '\n':
        word += next_char
        next_char = fd.read(1)

    next = fd.tell()
    is_last = next_char == '\n'

    val_skip = word.split(',')
    val = int(val_skip[0])
    if len(val_skip) > 1:
        skip = next + int(val_skip[1])
    else:
        skip = 0

    return Node(ptr, val, next, skip, is_last)


def get_node_wrapper(term, fd, ptr):
    """
    Automatically deciphers the correct parameterss for get_node depending on whether a list or posting_list is being
    processed.
    """
    if isinstance(term, list):
        return get_node(term, ptr)
    return get_node(fd, ptr)


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
        tokenized_query = parse_query(query)    # tokenize and process query
        postfix_queue = apply_shunting_algorithm(tokenized_query) # shunting algo to convert from infix to postfix order
        result = eval_postfix_queue(postfix_queue, global_dict, postings_fd)    # evaluate postfix token order
        result_list.append(result)

    # write results to output file
    with open(results_file, 'w') as r_file:
        for result in result_list:
            if isinstance(result, str):
                result = convert_term_to_postings(result, global_dict, postings_fd)
            result_str = ''
            for posting in result:
                result_str += str(posting) + ' '
            r_file.write(result_str.strip() + "\n")
    r_file.close()

    return


def parse_query(query):
    word_tokenized_query = word_tokenize(query)
    stemmed_tokens = []
    for token in word_tokenized_query:
        if (token not in operator) and (token not in bracket):  # exclude operators from stemming and case folding
            stemmed_tokens.append(stemmer.stem(token).lower())  # stem the token and convert to lowercase
        else:
            stemmed_tokens.append(token)

    duplicate_not_removed_tokens = remove_duplicate_not(stemmed_tokens)  # remove redundant NOT operators

    # consolidate 'x AND NOT y' operations to a simpler 'x ANDNOT y' operation. Likewise, 'NOT x and y' will become
    # 'y ANDNOT x'. This will make it easier to compute the NOT operation, as we no longer have to traverse the entire
    # universal docID list to separately compute 'NOT x' or 'NOT y'
    and_not_simplified_tokens = simplify_and_not(duplicate_not_removed_tokens)

    return and_not_simplified_tokens


def simplify_and_not(tokens):
    """
    Simplify 'x AND NOT y' operations to a simpler 'x ANDNOT y' operation.
    Likewise, 'NOT x and y' will become 'y ANDNOT x'. 'left ANDNOT right' takes the left list, and returns all elements
    that are not in the right list. Note that 'left ANDNOT right' order is enforced, where the right operand is the one
    that has the NOT boolean associated with it.
    """
    i = 0
    n = len(tokens)

    for token in tokens:
        if token == 'AND':
            # _ AND NOT y format
            if (i + 1 < n) and (tokens[i + 1] == 'NOT') and (i + 2 < n) and is_term(tokens[i + 2]):
                # 'NOT, x, and, NOT, y' format
                if (i - 1 >= 0) and (is_term(tokens[i - 1])) and (i - 2 >= 0) and (tokens[i - 2] == 'NOT'):
                    i += 1
                    continue
                # 'x, and, NOT, y' format, reformat to 'x, ANDNOT, _IGNORE, y'
                if (i - 1 >= 0) and (is_term(tokens[i - 1])) and (i - 2 >= 0) and (tokens[i - 2] != 'NOT'):
                    tokens[i] = 'ANDNOT'
                    tokens[i + 1] = '_IGNORE'  # used to remove redundant tokens after merging AND and NOT to ANDNOT
                    i += 1
                    continue
            # NOT x and _ format
            elif (i - 1 >= 0) and (is_term(tokens[i - 1])) and (i - 2 >= 0) and (tokens[i - 2] == 'NOT'):
                # NOT x and y format, reformat to 'y, ANDNOT, _IGNORE, x'
                if (i + 1 < n) and is_term(tokens[i + 1]):
                    tokens[i - 2] = tokens[i + 1]
                    tokens[i + 1] = tokens[i - 1]
                    tokens[i - 1] = 'ANDNOT'
                    tokens[i] = '_IGNORE'
                    i += 1
                    continue
            else:
                i += 1
        else:
            i += 1

    filter_ignore_token_list = []
    # remove _IGNORE symbols
    for token in tokens:
        if token != '_IGNORE':
            filter_ignore_token_list.append(token)

    return filter_ignore_token_list


def is_term(token):
    return (token not in operator) and (token not in bracket) and not token == '_IGNORE'


def remove_duplicate_not(tokens):
    """
    Remove duplicate NOT operators to avoid situations like 'NOT NOT NOT x', which is simply equivalent to 'NOT x'
    """
    not_count = 0
    duplicates_removed_list = []
    for token in tokens:
        if token == 'NOT':
            not_count += 1
            if not_count == 2:
                duplicates_removed_list.pop()
                not_count = 0
            else:
                duplicates_removed_list.append(token)
        else:
            not_count = 0
            duplicates_removed_list.append(token)

    return duplicates_removed_list


def apply_shunting_algorithm(tokens):
    """
    Convert infix token order to postfix.
    """
    term_queue = deque()  # append and popleft
    op_stack = []  # append and pop

    # process terms and boolean operators into a stack and queue respectively
    for token in tokens:
        if (token not in operator) and (token not in bracket):
            term_queue.append(token)
        elif token in operator:
            while op_stack and operator_precedence[op_stack[-1]] > operator_precedence[token]:
                popped_op = op_stack.pop()
                term_queue.append(popped_op)
            op_stack.append(token)
        elif token == '(':
            op_stack.append(token)
        elif token == ')':
            while op_stack and op_stack[-1] != '(':
                popped_op = op_stack.pop()
                term_queue.append(popped_op)
            op_stack.pop()

    # add all remaining stack elements to the queue
    while op_stack:
        term_queue.append(op_stack.pop())

    return term_queue


def eval_postfix_queue(postfix_queue, global_dict, postings_fd):
    """
    Evaluate the postfix expression
    """
    process_stack = []
    while postfix_queue:
        token = postfix_queue.popleft()
        if token not in operator:  # token is a term
            if token in global_dict:
                process_stack.append(token)
            else:
                # no such term in dictionary, so we append an empty list of postings
                process_stack.append([])

        elif token == 'ANDNOT':
            term2 = process_stack.pop()
            term1 = process_stack.pop()
            op_result = compute_andnot(term1, term2, global_dict, postings_fd)
            process_stack.append(op_result)

        elif token == 'NOT':
            term = process_stack.pop()
            op_result = compute_not(term, global_dict, postings_fd)
            process_stack.append(op_result)

        elif token == 'AND':
            term1 = process_stack.pop()
            term2 = process_stack.pop()
            op_result = compute_and(term1, term2, global_dict, postings_fd)
            process_stack.append(op_result)

        elif token == 'OR':
            term1 = process_stack.pop()
            term2 = process_stack.pop()
            op_result = compute_or(term1, term2, global_dict, postings_fd)
            process_stack.append(op_result)

    return process_stack.pop()


def compute_or(term1, term2, global_dict, postings_fd):
    # if both are empty lists, return empty list. If only either one is an empty list, simply return the other term.
    if isinstance(term1, list) and isinstance(term2, list) and not term1 and not term2:
        return []
    if isinstance(term1, list):
        if not term1:
            return term2
        else:
            term_id1, doc_freq1, ptr1 = None, len(term1), 0
    else:
        term_id1, doc_freq1, ptr1 = global_dict[term1]

    if isinstance(term2, list):
        if not term2:
            return term1
        else:
            term_id2, doc_freq2, ptr2 = None, len(term2), 0
    else:
        term_id2, doc_freq2, ptr2 = global_dict[term2]

    result = []
    node1 = get_node_wrapper(term1, postings_fd, ptr1)
    node2 = get_node_wrapper(term2, postings_fd, ptr2)
    while not node1.is_last and not node2.is_last:
        if node1.val > node2.val:
            result.append(node2.val)
            ptr2 = node2.next
            node2 = get_node_wrapper(term2, postings_fd, ptr2)

        elif node2.val > node1.val:
            result.append(node1.val)
            ptr1 = node1.next
            node1 = get_node_wrapper(term1, postings_fd, ptr1)

        elif node1.val == node2.val:
            result.append(node1.val)
            ptr1 = node1.next
            node1 = get_node_wrapper(term1, postings_fd, ptr1)
            ptr2 = node2.next
            node2 = get_node_wrapper(term2, postings_fd, ptr2)

    if node1.is_last and not node2.is_last:     # append remaining node 2 postings. also add in last node1 posting.
        while not node2.is_last:
            if node1.val > node2.val:
                result.append(node2.val)
                ptr2 = node2.next
                node2 = get_node_wrapper(term2, postings_fd, ptr2)
            else:
                if node1.val < node2.val:
                    result.append(node1.val)
                while not node2.is_last:
                    result.append(node2.val)
                    ptr2 = node2.next
                    node2 = get_node_wrapper(term2, postings_fd, ptr2)
                result.append(node2.val)
                return result
        result.append(node1.val)
        return result

    if node2.is_last and not node1.is_last:   # append remaining node1 postings. also add in last node2 posting.
        while not node1.is_last:
            if node2.val > node1.val:
                result.append(node1.val)
                ptr1 = node1.next
                node1 = get_node_wrapper(term1, postings_fd, ptr1)
            else:
                if node2.val < node1.val:
                    result.append(node2.val)
                while not node1.is_last:
                    result.append(node1.val)
                    ptr1 = node1.next
                    node1 = get_node_wrapper(term1, postings_fd, ptr1)
                result.append(node1.val)
                return result
        result.append(node2.val)
        return result

    if node1.is_last and node2.is_last:     # append last node1 and node2 postings.
        if node1.val < node2.val:
            result.append(node1.val)
            result.append(node2.val)
        elif node2.val < node1.val:
            result.append(node2.val)
            result.append(node1.val)
        else:
            result.append(node1.val)
        return result

    return result


def compute_not(term, global_dict, postings_fd):
    univ_id, univ_freq, univ_ptr = global_dict[UNIVERSAL]   # retrieve universal posting list of all docIDs
    if isinstance(term, list):
        if not term:
            return UNIVERSAL
        else:
            term_id, doc_freq1, term_ptr = None, len(term), 0
    else:
        term_id, doc_freq1, term_ptr = global_dict[term]

    result = []
    node_term = get_node_wrapper(term, postings_fd, term_ptr)
    node_univ = get_node_wrapper(UNIVERSAL, postings_fd, univ_ptr)
    while not node_term.is_last and not node_univ.is_last:
        if node_term.val > node_univ.val:
            result.append(node_univ.val)
            univ_ptr = node_univ.next
            node_univ = get_node_wrapper(UNIVERSAL, postings_fd, univ_ptr)
        elif node_univ.val > node_term.val:
            term_ptr = node_term.next
            node_term = get_node_wrapper(term, postings_fd, term_ptr)
        elif node_term.val == node_univ.val:
            term_ptr = node_term.next
            node_term = get_node_wrapper(term, postings_fd, term_ptr)
            univ_ptr = node_univ.next
            node_univ = get_node_wrapper(UNIVERSAL, postings_fd, univ_ptr)

    # term posting should have finished evaluation first since len of term posting should be <= len of universal_list
    final_node_term_val = node_term.val
    while not node_univ.is_last:
        if node_univ.val != final_node_term_val:
            result.append(node_univ.val)
        univ_ptr = node_univ.next
        node_univ = get_node_wrapper(UNIVERSAL, postings_fd, univ_ptr)

    if (node_term.is_last and node_univ.is_last) and (node_term.val != node_univ.val):
        result.append(node_univ.val)

    return result


def compute_and(term1, term2, global_dict, postings_fd):
    # if both terms are lists and either one is empty, return empty list
    if (isinstance(term1, list) and isinstance(term2, list)) and (not term1 or not term2):
        return []
    # if either term is an empty list, return empty list
    if (isinstance(term1, list) and not term1) or (isinstance(term2, list) and not term2):
        return []

    if isinstance(term1, list):
        term_id1, doc_freq1, ptr1 = None, len(term1), 0
    else:
        term_id1, doc_freq1, ptr1 = global_dict[term1]

    if isinstance(term2, list):
        term_id2, doc_freq2, ptr2 = None, len(term2), 0
    else:
        term_id2, doc_freq2, ptr2 = global_dict[term2]

    result = []
    node1 = get_node_wrapper(term1, postings_fd, ptr1)
    node2 = get_node_wrapper(term2, postings_fd, ptr2)
    while not node1.is_last or not node2.is_last:
        # increment ptr2
        if node1.val > node2.val:
            if node2.is_last:
                break
            if node2.skip != 0:
                skip_node2 = get_node_wrapper(term2, postings_fd, node2.skip)
                # safe to use skip
                if node1.val > skip_node2.val:
                    ptr2 = node2.skip
                    node2 = get_node_wrapper(term2, postings_fd, ptr2)
                    continue

            ptr2 = node2.next
            node2 = get_node_wrapper(term2, postings_fd, ptr2)

        # increment ptr1
        elif node1.val < node2.val:
            if node1.is_last:
                break
            if node1.skip != 0:
                skip_node1 = get_node_wrapper(term1, postings_fd, node1.skip)
                # safe to use skip
                if node2.val > skip_node1.val:
                    ptr1 = node1.skip
                    node1 = get_node_wrapper(term1, postings_fd, ptr1)
                    continue

            ptr1 = node1.next
            node1 = get_node_wrapper(term1, postings_fd, ptr1)

        # found a match!
        else:
            result.append(node1.val)
            if node1.is_last or node2.is_last:
                return result
            ptr1 = node1.next
            ptr2 = node2.next
            node1 = get_node_wrapper(term1, postings_fd, ptr1)
            node2 = get_node_wrapper(term2, postings_fd, ptr2)

        # no need to scan unfinished list for AND

    # edge case where both postings are at the last term, and the last terms are a match
    if (node1.is_last and node2.is_last) and (node1.val == node2.val):
        result.append(node1.val)

    return result


def compute_andnot(term1, term2, global_dict, postings_fd):
    # This refers to operations of the exact format 'term1 ANDNOT term2' which is equivalent to 'term1 AND NOT term2'

    # if term1 is an empty list, simply return an empty list as there are no terms to merge with
    if isinstance(term1, list):
        if not term1:
            return []
        else:
            term_id1, doc_freq1, ptr1 = None, len(term1), 0
    else:
        term_id1, doc_freq1, ptr1 = global_dict[term1]

    # if term2 is an empty list, simply return term1, as there are no postings in term2 that we need
    # to filter out from term1
    if isinstance(term2, list):
        if not term2:
            return term1
        else:
            term_id2, doc_freq2, ptr2 = None, len(term2), 0
    else:
        term_id2, doc_freq2, ptr2 = global_dict[term2]

    result = []
    node1 = get_node_wrapper(term1, postings_fd, ptr1)  # we want all node1's in final list
    node2 = get_node_wrapper(term2, postings_fd, ptr2)  # we do not want any node2's in final list
    while True:
        if node1.is_last and not node2.is_last:
            if node1.val < node2.val:
                result.append(node1.val)
                return result
            elif node1.val == node2.val:
                return result
            else:
                ptr2 = node2.next
                node2 = get_node_wrapper(term2, postings_fd, ptr2)
        elif node2.is_last and not node1.is_last:
            if node1.val != node2.val:
                result.append(node1.val)
            ptr1 = node1.next
            node1 = get_node_wrapper(term1, postings_fd, ptr1)

        elif node1.is_last and node2.is_last:
            if node1.val != node2.val:
                result.append(node1.val)
            return result

        elif node1.val > node2.val:
            if node2.skip != 0:
                skip_node2 = get_node_wrapper(term2, postings_fd, node2.skip)
                # safe to use skip
                if node1.val > skip_node2.val:
                    ptr2 = node2.skip
                    node2 = get_node_wrapper(term2, postings_fd, ptr2)
                    continue

            ptr2 = node2.next
            node2 = get_node_wrapper(term2, postings_fd, ptr2)

        elif node2.val > node1.val:
            result.append(node1.val)
            ptr1 = node1.next
            node1 = get_node_wrapper(term1, postings_fd, ptr1)

        elif node1.val == node2.val:
            ptr1 = node1.next
            node1 = get_node_wrapper(term1, postings_fd, ptr1)
            ptr2 = node2.next
            node2 = get_node_wrapper(term2, postings_fd, ptr2)

    return result


def convert_term_to_postings(term, global_dict, postings_fd):
    """
    Retrieve the posting_list of the given term and convert it to an array of postings.
    """
    term_id, doc_freq, ptr = global_dict[term]
    result = []
    node = get_node_wrapper(term, postings_fd, ptr)
    while not node.is_last:
        result.append(node.val)
        ptr = node.next
        node = get_node_wrapper(term, postings_fd, ptr)
    result.append(node.val)
    return result


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
