# WEAT
Implementation of WEAT as described by Caliskan, Bryson, and Narayanan in "Semantics derived automatically from language corpora contain human-like biases."

To use:

First, visit https://nlp.stanford.edu/projects/glove/ to download the pre-trained word embeddings. The embeddings in flowers_insects_pleasant_unpleasant_embedded.json use Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB). Download the file, unzip, and place the resulting txt file in the same directory. The file was ommitted due to Github space constraints.

Then run with:

python main.py --data_file_name DATA_FILE_NAME --embedded_data_file_name EMBEDDED_DATA_FILE_NAME --glove_file_name GLOVE_FILE_NAME --test TEST
where:
    DATA_FILE_NAME: name of target/attribute data file. In this example, flowers_insects_pleasant_unpleasant.json
    EMBEDDED_DATA_FILE_NAME: name of target/attribute embeddings file; if it does not exist it will be created. In this example, flowers_insects_pleasant_unpleasant_embedded.json
    GLOVE_FILE_NAME: name of GloVe embeddings file; required if EMBEDDED_DATA_FILE_NAME file does not exist
    TEST: WEAT or WEFAT