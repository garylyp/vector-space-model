This is the README file for A0189806M and A0225871B submission
Emails: e0325390@u.nus.edu; e0597734@u.nus.edu;

== Python Version ==

We're using Python Version 3.8.1 for this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps 
in your program, and discuss your experiments in general.  A few paragraphs 
are usually sufficient.

Section 1 details how indexing is down. Section 2 details the searching process.

1. INDEXING

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

2. SEARCH

During searching, we first preprocess each query term. If the query term exists in the dictionary, we keep 
it. Otherwise, we ignore that particular query term by deleting it. Note that since we have already
preprocessed the posting lists to contain the lnc value for each document, we do not have to do any
computation on documents in this stage. For each query, we then do the following:
1) Count the number of times each query term appears in the query. Store the frequency (integer), qtf.
2) For each unique query term, compute qtf' = 1 + log10(qtf)
3) Compute the vector length for the query, by taking sqrt(sum_of_squares_of_qtf')

At this point, for each unique query term, do the following:
  4) Retrieve the inverse document frequency (idf) from the dictionary.
  5) Compute and store the ltc score. Ltc score = qtf' * idf / vector_length
  6) Retrieve the relevant postings from dictionary
  7) For each docID's lnc value in the dictionary, compute and store the score = lnc_score * ltc_score
  8) If a score already exists for a  given docID, simply add the new score to the old score, and keep
     the total score.

Once all scores have been computed, for each relevant docID we insert the tuple (score, docID) into a heap.
We then retrieve the top 10 docIDs from the heap, based on score comparisons (with priority given to larger
scores). For equal scores, we compare based on docID values (with priority given to smaller docIDs).

If there are any empty queries, we correspondingly return an empty output for those queries.


EXPERIMENTS

We performed search on various types of inputs and verified if they return the expected documents.
The types of queries include
* Single-word queries ("russia", "caterpillar", "mexican") should fetch specific documents containing them
* Multi-word queries ("russia and moscow", "russia caterpillar") should fetch documents that contain both
  words or the most relevant documents for each of the word
* Typo, non-existent words ("mexixcan") should return no results as our program does not support fault
  tolerant search system yet

We also experimented with using the shortcut 'Faster Cosine' method of computing query weight, where 
instead of using the ltc approach, we assign a value of 1 to each term. Score computation would then
be equal to (doc_lnc * 1). What we observed was that the output list of docIDs would more or less be
the same, as compared to the lnc.ltc approach. The first 7 or so docIDs would be the same, but with
different ordering. The other 3 docIDs would be entirely different. We thus felt that this could
potentially be a much faster way of retrieving the top 10 most relevant docIDs.    


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

As per normal - based on correctness of code, documentation, accuracy and speed.

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>
