import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv


def get_lower_case_edges(edge,dict):
    L = []
    for e in edge:
        if (e[1] in dict) and (e[2] in dict):
            lan = int(e[0])
            w1 = dict[e[1]].lower()
            w2 = dict[e[2]].lower()
            L.append([w1,w2,lan])
    return L

def get_lower_case_edges_without_lan(edge,dict):
    L = []
    for e in edge:
        if (e[1] in dict) and (e[2] in dict):
            # lan = int(e[0])
            w1 = dict[e[1]].lower()
            w2 = dict[e[2]].lower()
            L.append([w1,w2])
    return L

conceptnet_clics3_edges = pd.read_csv('conceptnet_clics3_edges_v2.csv')
glove_edges = pd.read_csv('glove_6B_300d_edges_with_distance.csv')



concept_d = np.load("concept_dict_t.npy",allow_pickle = True)
concept_d = concept_d[0]
colex_edges = np.load("colex_edges.npy",allow_pickle = True)

edges_lower = get_lower_case_edges_without_lan(colex_edges,concept_d)
edges_lower_with_lan = get_lower_case_edges(colex_edges,concept_d)
result = []
en_rows = []
# 'conceptnet_clics3_edges_v2.csv'
# 'conceptnet_en_rows.csv'



# with open('conceptnet_clics3_edges_v2.csv',encoding="utf-8") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#
#     P = []
#     P_row = []
#
#     for row in csv_reader:
#         pair = row[:2]
#         if pair not in P:
#             P.append(pair)
#             P_row.append(row)
#         elif pair in P:
#             i = P.index(pair)
#             row_r = P_row[i]
#             print(row_r)
#             print(row)
#             G.append(pair)
#             G_row.append(row)

common_crawl_20k_dic = np.load("glove_common_crawl_nn20k_embedding_dict.npy",allow_pickle = True)
common_crawl_20k_dic= common_crawl_20k_dic[0]
keys = common_crawl_20k_dic.keys()
test = np.load("glove_common_crawl_20k_conceptnet_pair_relations.npy",allow_pickle = True)

with open('conceptnet_en_rows.csv',encoding="utf-8") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    c= 0
    t = 0

    for row in csv_reader:
        # print(t)



        #---------------find_edge_with_clics3-------------------
        r= row[0][7:-1]
        w1 = row[1][6:-1]
        w2 = row[2].split()[0][6:-2]

        if "/" in w1:
            w1 = w1.split('/')
            w1 = w1[0]

        if "/" in w2:
            w2 = w2.split('/')

            w2 = w2[0]

        # if len(w1) <= 3:
        #     print(w1)
        # if len(w2) <= 3:
        #     print(w2)


        if (w1 in keys) and (w2 in keys):
            result.append([w1, w2, r])
            c +=1
            print(c)


        # if [w1,w2] in edges_lower:
        #     result.append([w1, w2, r])
        # elif [w2,w1] in edges_lower:
        #     c+=1
        #     print(c)

            # result.append([w1,w2,r])
        # ----------------------------------------------------------



        # ------extract english rows when open the csv files for all languages-----------
        # if (row[1][:5]) == "/c/en":
        #     en_rows.append(row)
        #     c+= 1
            # print(c)
#         ---------------------------------



# save csv file
# with open('conceptnet_clics3_edges_v2_one_way.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(result)

#
# with open('glove_6B_300d_edges_with_distance.csv',encoding="utf-8") as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         print(row)



# for i in range(len(conceptnet_clics3_edges)):
#     w1 = conceptnet_clics3_edges["concept1"][i]
#     w2 = conceptnet_clics3_edges["concept2"][i]
#     rela = conceptnet_clics3_edges["relationship"][i]
#     for edge in edges_lower_with_lan:
#         if [w1,w2] == edge[0:2]:

#
# def combine_files(a,b):
#     new_rows = []
#     with open(a, encoding="utf-8") as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for row1 in csv_reader:
#
#             pair1 = row1[0:2]
#             pair1_r = pair1.reverse()
#             with open(b, encoding="utf-8") as csv_file:
#                 csv_reader = csv.reader(csv_file, delimiter=',')
#                 for row2 in csv_reader:
#                     pair2 = row2[1:3]
#                     if pair1 == pair2 or pair1_r == pair2:
#                         x = row2
#                         x.append(row1[2])
#                         new_rows.append(x)
#                         print(x)
#     return new_rows



def combine_files(a,b):
    overlap_rows = []
    non_rows = []
    with open(a, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row1 in csv_reader:
            c = 0

            pair1 = row1[1:3]
            pair1_r = pair1.reverse()
            with open(b, encoding="utf-8") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row2 in csv_reader:
                    pair2 = row2[0:2]
                    if pair1 == pair2 or pair1_r == pair2:
                        x = row1
                        x.append(row2[2])
                        # if x not in overlap_rows:
                        overlap_rows.append(x)

                        c += 1
                        # print(x)
            if c == 0:
                x = [int(row1[3]), row1[0], row1[1], row1[2]]
                # if x not in non_rows:
                non_rows.append(x)
    return overlap_rows, non_rows

def remove_reduandent(l):
    r = []
    for i in range(len(l)):
        print (i)
        if l[i] not in r:
            r.append(l[i])
    return r

#------------------------------remove redaundent pairs---------------------------------------------
# f1 = 'glove_twitter_27B_200d_edges_with_distance.csv'
# f2 = 'conceptnet_clics3_edges_v2.csv'
# overlap_rows, non_rows = combine_files(f1,f2)
# # sort by # lan
# non_rows.sort()
# non_rows.reverse()
#
# # overlap_rows format
# format = [["distance","concept1","concept2","#lan","relation"]]
# format.extend(overlap_rows)
# format = remove_reduandent(format)
#
# # non_overlap_rows format
# format2 = [["#lan","distance","concept1","concept2"]]
# format2.extend(non_rows)
# format2 = remove_reduandent(format2)
#---------------------------------------------------------------------------

# with open('v2.2_remove_redaundent/glove_twitter_relations_v2.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(format)
#
# with open('v2.2_remove_redaundent/glove_twitter_no_relation_v2.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(format2)
#

