#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import os, glob
import json
import linecache
import math

# Uncomment this line if your nltk package does not contain 'punkt'
# nltk.download('punkt') 

stemmer = nltk.stem.PorterStemmer()
universal_id_set = [] # list of all doc_id in the collection
UNIVERSAL = '_universal' # string representing the dummy term that exists in all docs (doc_freq = N)


def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')

    block_size = 1000  # Arbitrary. Can be higher if memory allows for it
    files = glob.glob(in_dir + "/*")
    if len(files) == 0:
        print('Document files not found. Check directory argument -i')
        return

    # Read the file in increasing order of doc_id to save complexity in merging step
    files.sort(key=lambda f: int(os.path.basename(f)))
    term_to_id = {}
    next_term_id = 0
    universal_id_set.extend([int(os.path.basename(file)) for file in files])

    block_id = 0
    block_total = math.ceil(len(files) / block_size)
    while block_id < block_total:
        print(f'processing block {block_id}')
        block = parse_block(files, block_id, block_size)
        block_index = bsbi_invert(block)
        next_term_id = write_block_to_disk(block_index, term_to_id, next_term_id, block_id)
        block_id += 1

    # Merge all blocks into one block (postings.txt)
    next_block_id = block_id
    block_files = glob.glob("block*")
    block_files.sort(key = lambda f : int(os.path.basename(f)[5:]))
    while len(block_files) > 1:
        next_block_id = merge_blocks(block_files, next_block_id)
        block_files = glob.glob("block*")
        block_files.sort(key = lambda f : int(os.path.basename(f)[5:]))

    if os.path.exists(out_postings):
        os.remove(out_postings)
    os.rename(block_files[0], out_postings)

    # read in postings.txt and insert skip pointers
    add_skip_pointers(out_postings)

    # Write dictionary (dictionary.txt)
    write_dictionary(term_to_id, out_dict, out_postings)


def read_file(filename):
    """
    Read the whole file as a string
    """
    f = open(filename, 'r')
    contents = f.read()
    f.close()
    return contents


def gen_tuples(contents, doc_id):
    """
    tokenize the entire file's content into a list of [(term, doc_id), (term, doc_id), ...]

    Apply pre-processing steps such as removal of punctuation, stemming
    """

    words = nltk.tokenize.word_tokenize(contents)

    # Removed punctuations
    words = [word for word in words if word.isalnum()]

    # Apply Porter Stemming (seems to have applied lowercase as well)
    words = [stemmer.stem(word) for word in words]

    # # remove duplicates
    # words = list(dict.fromkeys(words))
    tuples = get_doc_vector(words, doc_id)

    return tuples

def get_doc_vector(terms, doc_id):
    """
    Returns a list of tuples (term, doc_id, tf) for a given doc
    """
    # Get the term frequency for each term in the doc
    term_freq = {}
    for term in terms:
        if term in term_freq:
            term_freq[term] += 1
        else:
            term_freq[term] = 1
    
    # Convert each term_freq into log tf
    for term in term_freq:
        term_freq[term] = math.log10(term_freq[term]) + 1

    # Get vector length
    sum_of_squares = 0
    for term in term_freq:
        sum_of_squares += (term_freq[term] * term_freq[term])
    vector_length = sum_of_squares ** 0.5

    # Normalize vector
    for term in term_freq:
        term_freq[term] = term_freq[term] / vector_length

    # # Check normalization
    # sum_of_squares = 0
    # for term in term_freq:
    #     sum_of_squares += term_freq[term]**2
    # print(sum_of_squares)
    
    return [(term, doc_id, term_freq[term]) for term in term_freq]

def parse_block(files, block_id, block_size):
    """
    Reads a block of files and generate a list of (term, doc_id) tuples
    """
    block_of_tuples = []
    file_idx = block_id * block_size
    for i in range(file_idx, min(file_idx + block_size, len(files))):
        filename = files[i]
        contents = read_file(filename)
        doc_id = int(os.path.basename(filename))
        tuples = gen_tuples(contents, doc_id)
        block_of_tuples.extend(tuples)
    return block_of_tuples


def bsbi_invert(block):
    """
    Converts the list of (term, doc_id) tuples into an inverted index of the form
    {
        'term1' : [(1, tf), (5, tf), (9, tf), (20,tf), ...],
        'term2' : [(3, td), (9, tf), (30, tf)]
    }

    Where each posting list is sorted and contains no duplicates
    """
    temp_dict = {}
    for t in block:
        term, doc_id, tf = t
        if term in temp_dict:
            postings = temp_dict[term]
            postings[doc_id] = tf
        else:
            postings = { doc_id : tf}
            temp_dict[term] = postings

    # convert the postings (dict) into a postings list
    result_dict = {}
    for term in temp_dict:
        result_dict[term] = []
        for doc_id in temp_dict[term]:
            tf = temp_dict[term][doc_id]
            result_dict[term].append((doc_id, tf))

    return result_dict


def write_block_to_disk(index, term_to_id, next_term_id, block_id):
    """
    index: the inverted index for this block
    """
    block_name = "block{:03d}".format(block_id)
    f = open(block_name, 'w', newline='')

    term_ids = []
    id_to_term = {}
    for term in index:
        if term in term_to_id:
            term_id = term_to_id[term]
        else:
            term_id = int(next_term_id)
            term_to_id[term] = term_id
            next_term_id += 1

        term_ids.append(term_id)
        id_to_term[term_id] = term

    # Iterate the inverted index in increasing order of term_id
    # write to the temp posting file: term_id doc_id,tf doc_id,tf ...
    term_ids.sort() 
    for term_id in term_ids:
        term = id_to_term[term_id]
        new_posting = f'{term_id} '
        postings = index[term]
        new_posting += ' '.join([f'{p[0]},{p[1]}' for p in postings])
        f.write(new_posting)
        f.write('\n')

    f.close()
    return next_term_id


def merge_blocks(block_files, next_block_id):
    n = len(block_files)
    i = 0

    while i < n:
        k = min(2, n - i)
        print(f'merging block {int(block_files[i][5:])} and {int(block_files[i + k - 1][5:])} to {next_block_id}')
        merge_blocks_2_way(block_files, i, next_block_id)
        next_block_id += 1
        i += k

    return next_block_id


def merge_blocks_2_way(block_files, start_idx, next_block_id):
    block_name = "block{:03d}".format(next_block_id)
    f = open(block_name, 'w', newline='')

    lineno = [1] * 2
    if start_idx == len(block_files) - 1:
        while lineno[0] >= 0:
            new_posting = linecache.getline(block_files[start_idx], lineno[0])
            if not new_posting:
                lineno[0] = -1
                break
            lineno[0] += 1
            f.write(new_posting)
        os.remove(block_files[start_idx])
        f.close()
        return

    while True:
        line0 = linecache.getline(block_files[start_idx], lineno[0])
        if not line0:
            lineno[0] = -1
            break

        line1 = linecache.getline(block_files[start_idx + 1], lineno[1])
        if not line1:
            lineno[1] = -1
            break

        term_id_0 = get_term_id(line0)
        term_id_1 = get_term_id(line1)

        if term_id_0 < term_id_1:
            f.write(line0)
            lineno[0] += 1
        elif term_id_0 > term_id_1:
            f.write(line1)
            lineno[1] += 1
        else:
            term_id = term_id_0
            posting0 = get_posting_str(line0)
            posting1 = get_posting_str(line1)
            merged_posting = merge(posting0, posting1)
            k = -1
            for mp in merged_posting:
                cur_k = int(mp.split(',')[0])
                if cur_k <= k:
                    print("Incorrect merge order")
                    print(f'{term_id} k:{k} curr:{cur_k}')
                k = cur_k
                    
            posting_list = [str(term_id)] + merged_posting
            new_posting = ' '.join([x for x in posting_list])
            f.write(new_posting)
            f.write('\n')
            lineno[0] += 1
            lineno[1] += 1

    while lineno[0] >= 0:
        new_posting = linecache.getline(block_files[start_idx], lineno[0])
        if not new_posting:
            lineno[0] = -1
            break
        lineno[0] += 1
        f.write(new_posting)

    while lineno[1] >= 0:
        new_posting = linecache.getline(block_files[start_idx + 1], lineno[1])
        if not new_posting:
            lineno[1] = -1
            break
        lineno[1] += 1
        f.write(new_posting)

    os.remove(block_files[start_idx])
    os.remove(block_files[start_idx + 1])
    f.close()


def merge(x, y):
    # get the first doc_id of each list
    x_head = int(x[0].split(",")[0])
    y_head = int(y[0].split(",")[0])
    # append one list to the other as it is guaranteed
    # that the head of one list is greater than the tail 
    # of another list
    if x_head < y_head:
        return x + y
    else:
        return y + x


def get_term_id(posting_line):
    term_id_str = ''
    i = 0
    while posting_line[i] != ' ':
        term_id_str += posting_line[i]
        i += 1
    return int(term_id_str)


def get_posting(posting_line):
    posting_line = posting_line.strip()
    i = 0
    while posting_line[i] != ' ':
        i += 1

    i+=1 # skip past the whitespace
    postings = []
    for x in posting_line[i:].split(' '):
        if not x:
            continue
        if ',' in x:
            postings.append(int(x.split(',')[0]))
        else:
            postings.append(int(x))
    return postings


def get_posting_str(posting_line):
    posting_line = posting_line.strip()
    i = 0
    while posting_line[i] != ' ':
        i += 1
    i+=1 # skip past the whitespace
    return posting_line[i:].split(' ')


def add_skip_pointers(out_postings):
    temp_filename = 'postings_skip_ptrs.txt'
    temp_file = open(temp_filename, 'w', newline='')

    lineno = 1
    line = linecache.getline(out_postings, lineno)
    while line:
        augmented_line = augment_line(line)
        temp_file.write(augmented_line)
        temp_file.write('\n')
        lineno += 1
        line = linecache.getline(out_postings, lineno)

    universal_term_id = lineno - 1
    universal_id_line = f'{universal_term_id} ' + ' '.join([str(x) for x in universal_id_set])
    universal_id_aug_line = augment_line(universal_id_line)
    temp_file.write(universal_id_aug_line)
    temp_file.write('\n')

    temp_file.close()
    os.remove(out_postings)
    os.rename(temp_filename, out_postings)


def augment_line(line):
    """
    Add skip pointer to a posting list and return its string representation

    Skip pointer will be represented by an additional integer after the posting entry

    E.g. skip distance = 3
    ... 4,3 9 13 15,3 21 27 29,3 ...

    It indicate to the searcher to skip 3 whitespace to get to the next number
    """
    term_id = get_term_id(line)
    posting_list = get_posting_str(line)
    n = len(posting_list)
    k = math.floor(n ** 0.5)  # k is the skip distance
    new_line = f'{term_id} '
    for i in range(n):
        new_line += posting_list[i]
        if k > 1 and i % k == 0 and i + k < n:
            skip_offset = (k - 1) + sum([len(str(x)) for x in posting_list[i + 1:i + k]])
            new_line += f',{skip_offset}'
        if i < n - 1:
            new_line += ' '
    return new_line


def write_dictionary(term_to_id, out_dict, out_postings):
    """
    Generates the final dictionary and write it to 'dictionary.txt'
    dict : {
        'term1' : (term_id, idf, offset),
        'term2' : (term_id, idf, offset),
        ...
    }
    """
    final_dict = {}  # mapping of terms to (term_id, idf, offset)
    n = len(term_to_id) # number of terms
    term_to_id[UNIVERSAL] = n
    collection_size = len(universal_id_set)

    line_offsets_and_doc_freq = []
    offset = 0
    lineno = 1
    linecache.clearcache()
    line = linecache.getline(out_postings, lineno)
    while line:
        term_id = get_term_id(line)
        doc_freq = len(get_posting(line))
        idf = math.log(collection_size / doc_freq)
        line_offsets_and_doc_freq.append((term_id, idf, offset + get_term_id_len(term_id)))
        offset += len(line)
        lineno += 1
        line = linecache.getline(out_postings, lineno)

    for term in term_to_id:
        term_id = term_to_id[term]
        final_dict[term] = line_offsets_and_doc_freq[term_id]

    out_dict_file = open(out_dict, 'w')
    json.dump(final_dict, out_dict_file, sort_keys=False, indent=2)
    out_dict_file.close()


def get_term_id_len(term_id):
    """
    Returns the length (number of bytes) of the term_id when in str representation
    '1 ' --> 2
    '24 ' --> 3
    """
    return len(str(term_id)) + 1


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i':  # input directory
        input_directory = a
    elif o == '-d':  # dictionary file
        output_file_dictionary = a
    elif o == '-p':  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
