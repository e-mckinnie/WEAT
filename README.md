# WEAT
Implementation of WEAT (and WEFAT) as described by Caliskan, Bryson, and Narayanan in "Semantics derived automatically from language corpora contain human-like biases" (Science, 2017, DOI: 10.1126/science.aal4230).

To use:

First, visit https://nlp.stanford.edu/projects/glove/ to download the pre-trained word embeddings. The embeddings in weat_1.json and wefat_1.json use Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB). Download the file, unzip, and place the resulting txt file in the same directory. The file was ommitted in this repository due to Github space constraints.

Next install requirements.txt with pip install.

Then run with:

python main.py --data_file_name DATA_FILE_NAME --embedded_data_file_name EMBEDDED_DATA_FILE_NAME --glove_file_name GLOVE_FILE_NAME --wefat_association_file_name WEFAT_ASSOCIATION_FILE_NAME --test TEST --iterations N --distribution_type DISTRIBUTION_TYPE

where: <br>
    DATA_FILE_NAME: name of target/attribute data file. In this example, weat_1.json or wefat_1.json <br>
    EMBEDDED_DATA_FILE_NAME: name of target/attribute embeddings file; if it does not exist it will be created. In this example, weat_1_embedded.json or wefat_1_embedded.json <br>
    GLOVE_FILE_NAME: name of GloVe embeddings file; required if EMBEDDED_DATA_FILE_NAME file does not exist <br>
    WEFAT_ASSOCIATION_FILE_NAME: name of file that maps target to other statistic. In this example, wefat_1_percentage_women.json maps profession (target) to % women <br>
    TEST: WEAT or WEFAT <br>
    N: number of iterations to calculate p-value <br>
    DISTRIBUTION_TYPE: type of distribution to compute p-value<br>
    
## WEAT
Displays the effect size and the p-value.

## WEFAT
Displays a graph of test statistic vs other statistic (e.g. percentage of women in an occupation) and Pearson's correlation coefficient.
 if wefat_assocation_file_name is specified. Otherwise displays effect size and p-value for each target word.

### Resources
Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334), 183–186. https://doi.org/10.1126/science.aal4230<br>
<br>
Pennington, J. (2014). GloVe: Global Vectors for Word Representation. Stanford.edu. https://nlp.stanford.edu/projects/glove/<br>
<br>
Guide to Using Pre-trained Word Embeddings in NLP. (2021, June 1). Paperspace Blog. https://blog.paperspace.com/pre-trained-word-embeddings-natural-language-processing/#using-glove-word-embeddings<br>
<br>
U.S. Bureau of Labor Statistics. (2019, January 18). Employed persons by detailed occupation, sex, race, and Hispanic or Latino ethnicity. Bls.gov. https://www.bls.gov/cps/cpsaat11.htm<br>
<br>
Toney-Wails, A., & Caliskan, A. (2021). ValNorm Quantifies Semantics to Reveal Consistent Valence Biases Across Languages and Over Centuries. ArXiv:2006.03950 [Cs]. https://arxiv.org/abs/2006.03950<br>
<br>
Caliskan, A. (2017). Replication Data for: WEFAT and WEAT(UNF:6:C9yaa+UeGfFmtuHz734iKw==, V2). [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/DX4VWP<br>
