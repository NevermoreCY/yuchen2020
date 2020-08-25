import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv

def dict_to_list_lower(dict):
    l = []
    for key in dict:
        l.append(dict[key].lower())

    return l

def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

# def find_euclidean_for_pair(x,y):
#     return spatial.distance.euclidean(embeddings_dict[pair[0]], embeddings_dict[pair[1]])

def get_euclidean_for_edges(edges, embeddings_dict):
    l = []
    for edge in edges:
        w1 = edge[0]
        w2 = edge[1]
        lan = edge[2]
        # print(w1,w2)
        if w1 in embeddings_dict and w2 in embeddings_dict and w1 != w2:

            v1 = embeddings_dict[w1]
            v2 = embeddings_dict[w2]
            d = spatial.distance.euclidean(v1,v2)
            if [d,w1,w2] not in l:
                l.append([d,w1,w2,lan])
    return l




    

def get_lower_case_edges(edge,dict):
    L = []
    for e in edge:
        if (e[1] in dict) and (e[2] in dict):
            lan = int(e[0])
            w1 = dict[e[1]].lower()
            w2 = dict[e[2]].lower()
            L.append([w1,w2,lan])
    return L


def divide_by_threshold(t,list):
    for i in range(len(list)):
        n = list[i][0]
        if n > t:
            return list[:i], list[i:]
    return None, None





noisy_value = []

concept_d = np.load("concept_dict_t.npy",allow_pickle = True)
concept_d = concept_d[0]
concept_l = dict_to_list_lower(concept_d)
colex_edges = np.load("colex_edges.npy",allow_pickle = True)
edges_lower = get_lower_case_edges(colex_edges,concept_d)
total_common_crawl = 2195989
extract_sample_size = 20000
size_count = 0
#---------------To extract pre-trained embeddings--------------------------
embeddings_dict = {}
with open("glove_840B_300d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        if len(values) != 301:
            print(len(values))
            print(values)
            noisy_value.append(values)
        else:
            word = values[0]
            if word in concept_l:
                # print(word)
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
            elif np.random.randint(0,total_common_crawl) <= extract_sample_size:
                size_count+=1
                print(size_count)
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

#----------------------------------------------------------------------------
test = np.load("glove_common_crawl_nn20k_embedding_dict.npy",allow_pickle = True)
common_crawl_embeddings_dict = np.load("glove_clics3_embedding_dict.npy",allow_pickle = True)
common_crawl_embeddings_dict = common_crawl_embeddings_dict[0]
glove_6B_300d_embedding_dict = np.load("glove_6B_300d_embedding_dict.npy",allow_pickle = True)
glove_6B_300d_embedding_dict = glove_6B_300d_embedding_dict[0]
glove_twitter_27B_200d_embedding_dict = np.load("glove_twitter_27B_200d_embedding_dict.npy",allow_pickle = True)
glove_twitter_27B_200d_embedding_dict = glove_twitter_27B_200d_embedding_dict[0]

common_crawl = get_euclidean_for_edges(edges_lower, common_crawl_embeddings_dict)
common_crawl.sort()
threshold_common_crawl = (common_crawl[0][0] + common_crawl[-1][0])*0.5
sim_common_crawl,non_common_crawl = divide_by_threshold(threshold_common_crawl,common_crawl)

# with open('glove_common_crawl_840B_300d_edges_with_distance.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(common_crawl)
# #
# with open('glove_common_crawl_840B_300d_edges_with_distance_sim.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(sim_common_crawl)
#
# with open('glove_common_crawl_840B_300d_edges_with_distance.csv_non_sim.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(non_common_crawl)

# glove_840B_300d = [["euclidean distance", "concept1", "concept2"]]
# glove_840B_300d.extend(edges_with_d)

glove_6B_300d = get_euclidean_for_edges(edges_lower, glove_6B_300d_embedding_dict)
glove_6B_300d.sort()
threshold_wiki = (glove_6B_300d[0][0] + glove_6B_300d[-1][0])*0.5
sim_wiki,non_sim_wiki = divide_by_threshold(threshold_wiki,glove_6B_300d)

# with open('glove_6B_300d_edges_with_distance.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(glove_6B_300d)
#
# with open('glove_6B_300d_edges_with_distance_sim.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(sim_wiki)
#
# with open('glove_6B_300d_edges_with_distance.csv_non_sim.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(non_sim_wiki)



# with open('glove_840B_300d_edges_with_distance.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(glove_840B_300d)


# with open('glove_840B_300d_edges_with_distance.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(glove_840B_300d)


glove_twitter_27B_200d = get_euclidean_for_edges(edges_lower, glove_twitter_27B_200d_embedding_dict)
glove_twitter_27B_200d.sort()
threshold_twitter = (glove_twitter_27B_200d[0][0] + glove_twitter_27B_200d[-1][0])*0.5
sim_twitter,non_sim_twitter = divide_by_threshold(threshold_twitter,glove_twitter_27B_200d)

# with open('glove_twitter_27B_200d_edges_with_distance.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(glove_twitter_27B_200d)
#
# with open('glove_twitter_27B_200d_edges_with_distance_sim.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(sim_twitter)
#
# with open('glove_twitter_27B_200d_edges_with_distance.csv_non_sim.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(non_sim_twitter)




# np.save("glove_clics3_embedding_dict",[embeddings_dict],allow_pickle=True)

# n = 0.
# for row in common_crawl:
#     n += row[3]


