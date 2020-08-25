import networkx as nx
import numpy as np
import csv

# e = ('2304', '2312')

def extract_colexification_dictionary(G):
    dict = {}
    for i in G.node:
        key = G.node[i]['Gloss']
        entry = []
        for edge in G[i]:
            entry.append(G.node[edge]['Gloss'])
        dict[key] = entry
    return dict

def extract_gloss_with_id(G):
    result = []
    for i in G.node:
        gloss = G.node[i]['Gloss']
        id = i
        result.append([id,gloss])
    return result

def extract_edges_with_language_weight(G):
    result = []
    for edge in G.edges:
        w1 = G.node[edge[0]]['Gloss']
        w2 = G.node[edge[1]]['Gloss']
        lan_weight = G.edges[edge]['LanguageWeight']
        result.append([lan_weight,w1,w2])
    return result


def extract_edge_id_with_language_weight(G):
    result = []
    for edge in G.edges:
        w1 = edge[0]
        w2 = edge[1]
        lan_weight = G.edges[edge]['LanguageWeight']
        result.append([lan_weight,w1,w2])
    return result


# read the gml one created by README.md in clics^3
G = nx.read_gml('network-2-languages.gml')




# extract cvs pairs
# rows = extract_edges_with_language_weight(G)
# rows.sort()
# rows.append(['#languages that colexify these two concepts','concept1','concept2'])
# rows.reverse()
#
# with open('network-1-language-lanugageWeight.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(rows)


# convert to dict of lists (one prefer to use dictionary instead of GML structure but feel free to just use gml)
# dict = [extract_colexification_dictionary(G)]

#
# id = extract_gloss_with_id(G)


# save to a np file so can be used in python 2.7 with histowords.
# np.save("clics_1_languages",dict,allow_pickle=True)
# np.save("concept_list",[id],allow_pickle=True)

# np.save("colex_edges",rows,allow_pickle=True)


#-------------------------7.23.2020-------------------------
# rows = extract_edge_id_with_language_weight(G)
#
# #test
# a = np.array([[1,10,100],[2,8,200],[3,5,150]])
#
# a_sorted = a[np.argsort(a[:,1])]
