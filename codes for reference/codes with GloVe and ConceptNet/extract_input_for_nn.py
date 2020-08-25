import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv
import pickle
# np.save("glove_clics3_embedding_dict",[embeddings_dict],allow_pickle=True)

def get_relations(n_hot,format):
    r = []
    for i in range(len(n_hot)):
        if n_hot[i] > 0:
            r.append(format[i])
    return r


common_crawl_embeddings_dict = np.load("glove_clics3_embedding_dict.npy",allow_pickle = True)
common_crawl_embeddings_dict = common_crawl_embeddings_dict[0]
glove_6B_300d_embedding_dict = np.load("glove_6B_300d_embedding_dict.npy",allow_pickle = True)
glove_6B_300d_embedding_dict = glove_6B_300d_embedding_dict[0]
glove_twitter_27B_200d_embedding_dict = np.load("glove_twitter_27B_200d_embedding_dict.npy",allow_pickle = True)
glove_twitter_27B_200d_embedding_dict = glove_twitter_27B_200d_embedding_dict[0]

common_crawl_20k_dic = np.load("glove_common_crawl_nn20k_embedding_dict.npy",allow_pickle = True)
common_crawl_20k_dic= common_crawl_20k_dic[0]
test = np.load("glove_common_crawl_20k_conceptnet_pair_relations.npy",allow_pickle = True)

format = np.load("label_format.npy",allow_pickle=True)


cce = 'conceptnet_clics3_edges_v2.csv'
input_cc = "v2.2_remove_redaundent/has_relation/glove_common_crawl_relations_v2.csv"

# combine the relations
# with open(cce,encoding="utf-8") as csv_file:
#     pairs = []
#     rows = []
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         p = row[0:2]
#         if p not in pairs:
#             pairs.append(p)
#             rows.append(row)
#         if p in pairs:
#             i = pairs.index(p)
#             rows[i].append(row[2])
#
#
#
#         print(row)





# extraction nn input and nn label
#-------------------------------------------------------------------------------
# pairs = np.load("nn_data/pairs_with_relations.npy",allow_pickle=True )
# dict = common_crawl_embeddings_dict
# pairs = test
# dict = common_crawl_20k_dic
# input_nn = []
# label_nn = []
#
# for row in pairs:
#     pair = row[0:2]
#     relations = row[2:]
#
#     print("Pair", pair)
#     print("Relations", relations)
#
#     # use pair to extract embedding vectors (Input for nn).
#     v1 = dict[pair[0]]
#     v2 = dict[pair[1]]
#     v = np.hstack((v1, v2))
#     input_nn.append(v)
#
#     # use Relations to make N-hot vector target label (Output for nn).
#
#     label = np.zeros(format.size)
#     for i in range(format.size):
#         if format[i] in relations:
#             label[i] = 1.
#     label_nn.append(label)
#
#
# input_nn = np.array(input_nn)
# label_nn = np.array(label_nn)
#
# print(input_nn.shape)
# print(label_nn.shape)

#-------------------------------------------------------------------------------
#
# np.save("nn_data/wiki_input_vector",input_nn,allow_pickle=True)
# np.save("nn_data/wiki_label_vector",label_nn,allow_pickle=True)

# np.save("nn_data/common_crawl_20k_input_vector",input_nn,allow_pickle=True)
# np.save("nn_data/common_crawl_20k_label_vector",label_nn,allow_pickle=True)


# extract pairs with no relation
#-------------------------------------------------------------------------------------------------------
# glove_cc_no_relation = "v2.2_remove_redaundent/no_relation/glove_common_crawl_no_relation_v2.csv"
# dict = common_crawl_embeddings_dict
# with open(glove_cc_no_relation,encoding="utf-8") as csv_file:
#     vectors = []
#     pairs = []
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#             print(row)
#             w1 = row[2]
#             if w1 != 'concept1':
#                 lan = row[0]
#                 w2 = row[3]
#                 pairs.append([lan,w1,w2])
#                 # get
#                 v1 = dict[w1]
#                 v2 = dict[w2]
#                 v = np.hstack((v1, v2))
#                 vectors.append(v)
                # p = row[0:2]
                # if p not in pairs:
                #     pairs.append(p)
                #     rows.append(row)
                # if p in pairs:
                #     i = pairs.index(p)
                #     rows[i].append(row[2])
#-----------------------------------------------------------------------


# np.save("nn_data/common_crawl_test_vector",vectors,allow_pickle=True)
   #   save
# dbfile = open('nn_data/test_pairs', 'wb')
# pickle.dump(pairs , dbfile)
# dbfile.close()

#--------------------------------------output csv file for nn prediction--------------------------------------------------
# load labels
dbfile = open('nn_data/test_pairs', 'rb')
pairs = pickle.load(dbfile)
dbfile.close()
#
dbfile = open('nn_data/final_output_20k_k_5_t_2', 'rb')
result = pickle.load(dbfile)
dbfile.close()
#
# # combine pairs and output from nn
#
Final = []
for i in range(len(result)):
    relation = get_relations(result[i], format)
    pairs[i].extend(relation)

with open('nn_data/prediction/final_20k_5_fold_t_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(pairs)


#--------------------------------------8.17--------------------------------------------------



