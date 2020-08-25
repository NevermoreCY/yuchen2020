# yuchen2020
# semantic change and coleixfictions

Output result:
The output results conatins all the output plots and files mentioned in the paper. 

codes for reference:
The codes for reference file conatins codes one used for this project. Some intermediate samples are saved in the raw samples files since it might need to be loaded in the code.

codes with histoword：
The codes under file "codes with histoword" in github repository need to be run with HistWords data. And should be put under the location "histwords-master/histwords-master".  HistWords data can be downloaded here: https://nlp.stanford.edu/projects/histwords/
WOLD.py is a reference code to plot semantice change histogram for 22 categories in WOLD. As well as the p value and p coefficient for combained data.
colex_neighbor_predict.py is a refrence code to predict what kind of colexification words can have by it's nearest k neighbors in histoword embeddings.
clics3_with_histowrods.py relates coleixfication pairs in clics3 with histwords embeddings. This is the eariler version and later histwords is replaced by GloVe emebeddings.
export_csv.py is used to output csv files for coelxification pairs.

codes with IDS:
The codes under file "codes with IDS" in github repository need to be run with pre-processed IDS data. And should be put under the location "codes for replication".  The IDS data one used are pre-extracted in paper http://www.cs.toronto.edu/~yangxu/xu_et_al_2020_colexification_preprint.pdf.
extract_ids_info.py: Extracts and calculates the colexifiaction counts for different categories.

codes with clics3:
The codes under file "codes with clics3" in github repository need to be run with clics3 data. And should be put under the location "clics3-master/graphs/data".  The clics data can be found in https://clics.clld.org/.
extract_network.py: extract conlexification pairs from the gml file.

codes with GloVe and ConceptNet:
The codes under file "codes with clics3" in github repository need to be run with GloVe data. And should be put under the location "GloVe-master/eval/python".  The GloVe data can be found in https://nlp.stanford.edu/projects/glove/.
extract_conceptnet.py : extract coneptnet relations for clics3 colexification pairs .
extract_glove.py : extract clics3 colexification pairs from glove.
extract_input_for_nn.py: extract input data and label in proper format.
neural_net.py: codes for the neural network model that is used to predict the relations from ConceptNet.
evaluate_result.py: evaluate the prediction of the neural network.
extract_all_glove_embeddings.py : furthur steps that extract non colexification pairs from glove.
