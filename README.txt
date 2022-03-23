This is the README file for A0189806M and A0225871B submission
Emails: e0325390@u.nus.edu; e0597734@u.nus.edu;

== Python Version ==

We're using Python Version 3.8.1 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

INDEXING

The BSBI algorithm to generate and merge blocks was preserved. To support a search system based on
a vector space model, the term frequencies and document frequencies of the terms must be calculated
and stored in the dictionary.txt and postings.txt. The tf and idf values are based on the lnc.ltc 
format. tf for each term in a document is computed as follows. For each term in the document
1) Count the term's frequency (integer), tf
2) Compute tf' = 1 + log10(tf)
3) Compute the vector length for this document by taking sqrt(sum_of_squares_of_tf')
4) Compute and store in the posting list for this term normalized_tf = tf' / vector_length
4.1) Each posting in the posting list is a comma separated string of "doc_id,normalized_tf"
5) The posting list are written into postings.txt and consists of space-separated postings:
"""
term_id_x doc1,normalized_tf1 doc2,normalized_tf2 ...
term_id_y doc1,normalized_tf1 doc2,normalized_tf2 ...
"""

The idf value for every term is calculated as follows
1) doc_freq = length of posting list
2) idf = log(N/doc_freq) where N is the total number of documents in the collection
3) The idf values are stored in the dictionary.txt, which now contains the following 
   information for each term
"""
term_1 : [
    term_id,
    idf,
    offset_to_posting_list // synonymous of a pointer
],
term_2 : [
    term_id,
    idf,
    offset_to_posting_list
],
"""

Note that skip pointers are not used but still preserved for potential future use.

SEARCH

todo...

EXPERIMENTS

We performed search on various types of inputs and verified if they return the expected documents.
The types of queries include
* Single-word queries ("russia", "caterpillar", "mexican") should fetch specific documents containing them
* Multi-word queries ("russia and moscow", "russia caterpillar") should fetch documents that contain both
  words or the most relevant documents for each of the word
* Typo, non-existent words ("mexixcan") should return no results as our program does not support fault
  tolerant search system yet
    


== Files included with this submission ==

1. README.txt: A text file containing information relevant to indexing and querying processes.
2. index.py: Responsible for indexing for vector space model using lnc.ltc format.
3. search.py: Responsible for searching and ranking of documents based on their cosine scores.
4. dictionary.txt: Dictionary of terms with term_id, idf, and byte_offset to posting.
5. postings.txt: Posting Lists. Each posting list contains term_id, postings, and skip pointers.
6. ESSAY.txt: Discussion of essay questions

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[X] We, A0189806M and A0225871B, certify that we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, we
expressly vow that we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.  

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>
