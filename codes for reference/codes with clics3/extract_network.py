import networkx as nx
import numpy as np
import csv

# e = ('2304', '2312')

def extract_colexification_dictionary(G):
    """extract colexification pairs in form of dictionary"""
    dict = {}
    for i in G.node:
        key = G.node[i]['Gloss']
        entry = []
        for edge in G[i]:
            entry.append(G.node[edge]['Gloss'])
        dict[key] = entry
    return dict

def extract_gloss_with_id(G):
    """extract all meanings and its id in G.node"""
    result = []
    for i in G.node:
        gloss = G.node[i]['Gloss']
        id = i
        result.append([id,gloss])
    return result

def extract_edges_with_language_weight(G):
    """extract colexification pairs with it's language weight"""
    result = []
    for edge in G.edges:
        w1 = G.node[edge[0]]['Gloss']
        w2 = G.node[edge[1]]['Gloss']
        lan_weight = G.edges[edge]['LanguageWeight']
        result.append([lan_weight,w1,w2])
    return result


def extract_edge_id_with_language_weight(G):
    """extract id of pairs with it's language weight"""
    result = []
    for edge in G.edges:
        w1 = edge[0]
        w2 = edge[1]
        lan_weight = G.edges[edge]['LanguageWeight']
        result.append([lan_weight,w1,w2])
    return result


# read the gml file generated in the work flow. Since the GML file is too larger,
# it can't be upload to github. One can generate the gml file by oneself or download a
# pre generated one from clics3.

G = nx.read_gml('network-2-languages.gml')




## extract colexification pairs with language weight

# rows = extract_edges_with_language_weight(G)
# rows.sort()
# rows.append(['#languages that colexify these two concepts','concept1','concept2'])
# rows.reverse()

## save rows to csv files

# with open('network-1-language-lanugageWeight.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(rows)




# convert to dict of lists (the csv form above is more useful, so ignore this one)
# dict = [extract_colexification_dictionary(G)]

# id = extract_gloss_with_id(G)


# save to a np file so can be used in python 2.7 with histowords.
# np.save("clics_1_languages",dict,allow_pickle=True)
# np.save("concept_list",[id],allow_pickle=True)

# np.save("colex_edges",rows,allow_pickle=True)

#-------------------------test---------------------
# rows = extract_edge_id_with_language_weight(G)
# a_sorted = a[np.argsort(a[:,1])]
