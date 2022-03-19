index:
	python3 index.py -i 'reuters/training' -d 'dictionary.txt' -p 'postings.txt'
	# windows:
	# python3 index.py -i reuters/training -d dictionary.txt -p postings.txt

search:
	python3 search.py -d 'dictionary.txt' -p 'postings.txt' -q 'queries.txt' -o 'output.txt'
	# windows:
	# python3 search.py -d dictionary.txt -p postings.txt -q queries.txt -o output.txt
