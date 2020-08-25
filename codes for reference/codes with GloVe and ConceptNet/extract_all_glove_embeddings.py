import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv

alphabet = "abcdefghijklmnopqrstuvwxyz"

# ---------------To extract pre-trained embeddings--------------------------
embeddings_dict = {}
with open("glove_840B_300d.txt", 'r', encoding="utf-8") as f:
    c = 0
    for line in f:
        values = line.split()
        if len(values) == 301:
            word = values[0]
            # print(word)
            if word.lower()[0] in alphabet and len(word)<= 20:
                c += 1
                print(c)
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word.lower()] = vector
print(c)
# ----------------------------------------------------------------------------

# with open('conceptnet_en_rows.csv',encoding="utf-8") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     c= 0
#     t = 0
#
#     for row in csv_reader:
        # print(t)



        #---------------find_edge_with_clics3-------------------
        # r= row[0][7:-1]
        # w1 = row[1][6:-1]
        # w2 = row[2].split()[0][6:-2]
        #
        # if "/" in w1:
        #     w1 = w1.split('/')
        #     w1 = w1[0]
        #
        # if "/" in w2:
        #     w2 = w2.split('/')
        #
        #     w2 = w2[0]

        # if len(w1) <= 3:
        #     print(w1)
        # if len(w2) <= 3:
        #     print(w2)

        # if [w1,w2] in edges_lower:
        #     result.append([w1, w2, r])
        # elif [w2,w1] in edges_lower:
        #     c+=1
        #     print(c)

            # result.append([w1,w2,r])
        # ----------------------------------------------------------



        # ------extract english rows-----------
        # if (row[1][:5]) == "/c/en":
        #     en_rows.append(row)
        #     c+= 1
            # print(c)
#         ---------------------------------