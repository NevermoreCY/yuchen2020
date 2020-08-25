import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import scipy
import pandas as pd
from representations.sequentialembedding import SequentialEmbedding
import csv


def transfer_to_one_word(dict):
    transferable = {}
    filtered = []
    for key in dict:
        s = key.split()
        remove = []
        for item in s:
            if ("(" in item) or (")" in item):
                remove.append(item)
        for r in remove:
            s.remove(r)
        if "OR" in s:
            transferable[key] = s[0]
        elif len(s) == 1:
            transferable[key] = s[0]
        else:
            filtered.append(key)
    return transferable,filtered


def transfer_word_list(l):
    result = {}
    for p in l:
        id = p[0]
        concept = p[1]
        s = concept.split()
        remove = []
        for item in s:
            if ("(" in item) or (")" in item):
                remove.append(item)
        for r in remove:
            s.remove(r)
        # if "OR" in s:
        #     transferable[key] = s[0]
        if len(s) == 1:
            result[id] = s[0]

    return result


same = []
def dict_to_one_word(dict, transferable):
    result = {}
    count = 0
    for key in dict:
        if key in transferable:
            result_key = transferable[key]
            result_entry = []
            for word in dict[key]:
                if word in transferable:
                    result_entry.append(transferable[word])
            if result_key in result:
                count += 1
                same.append(key)
                for i in result_entry:
                    if i not in result[result_key]:
                        result[result_key].append(i)
            else:
                result[result_key] = result_entry
    return result, count

dont_have = []
#253

def calcualte_overlap_rate(embds,dict_one_word,year,n=100):
    rate = []
    embds_y = embds.embeds[year]

    for word in dict_one_word:
        neigh = embds_y.closest(word.lower(), n + 1)
        if neigh == "we don't have this word":
            dont_have.append(word.lower())
        else:
            count = 0.
            print("word:", word)
            print("dic:", dict_one_word[word])
            print("neigh",neigh)
            neigh_words = np.array(neigh)[:,1]
            for colex in dict_one_word[word]:
                if colex.lower() in neigh_words:
                    count += 1.
            if len(dict_one_word[word]) != 0:
                print("rate count", count )
                rate.append(1.0 * count / len(dict_one_word[word]))

        # else:
        #      rate.append(0)
    return rate



def get_one_word_colex_edges(embds,colex_edge,concept_dict,year):
    result = []
    embds_y = embds.embeds[year]
    for edge in colex_edge:
        lan = edge[0]
        id1 = edge[1]
        id2 = edge[2]

        if id1 in concept_dict and id2 in concept_dict:
            print("True")
            w1 = concept_dict[id1].lower()
            w2 = concept_dict[id2].lower()
            s = embds_y.similarity(w1,w2 )
            print("w1,w2 = ",w1,w2)
            print("S is ",s)
            if (s > 0) and (s < 0.99):
                result.append([w1,w2,lan,s])
    return result


def sort_by_column(a,c_id):
    column = a[:,c_id]
    n = []
    for i in column:
        n.append(float(i))
    n = np.array(n)
    r = a[np.argsort(n)]

    return r


def cut_by_sim_threshold(t,sorted_l):
    sim_l = sorted_l[:,3]
    for i in range(len(sim_l)):
        x = float(sim_l[i])
        if x >= t:
            return sorted_l[0:i] , sorted_l[i:]
    return None, None












# load clics3 data
# clics3_dict = np.load("clics_2_languages.npy",allow_pickle = True)
# clics3_dict = np.load("clics_1_languages.npy",allow_pickle = True)
# clics3_dict = clics3_dict[0]




# non_word counts:
# 891 total
# 532 remove bracket
# 427 remove OR (105 words are first word in "OR" clause)

# transfer the clics3 dictionary to concepts expressed in only 1 english word.
# transferable, filtered = transfer_to_one_word(clics3_dict)
# count is for the reduandent word
# dict_one_word,count = dict_to_one_word(clics3_dict, transferable)

# np.save("clics_2_dict",[dict_one_word],allow_pickle=True)
#get word histwords embeddings
embds = SequentialEmbedding.load("embeddings/eng-all_sgns/sgns", range(1890, 2000, 10))
year_s = 1890
year_e = 1990


# rate = calcualte_overlap_rate(embds,dict_one_word,1990,n=100)

# clics_2_languages
#1990 rate mean = 0.18535519972
#1890 rate mean = 0.21535519972

#clics_1_languages
# 1990 rate mean = 0.08764466776061722
# 1890 rate mean = 0.09757857460610882

#----------------7.23.2020-----------------------------------
# untransfered word list


concept_list = np.load("concept_list.npy",allow_pickle = True)
concept_list = concept_list[0]

concept_dict_t = transfer_word_list(concept_list)
colex_edges = np.load("colex_edges.npy",allow_pickle = True)
colex_edges_sim = get_one_word_colex_edges(embds, colex_edges, concept_dict_t, year_s)
colex_edges_sim = np.array(colex_edges_sim)

colex_edges_sort_by_lan = sort_by_column(colex_edges_sim,2)
colex_edges_sort_by_sim = sort_by_column(colex_edges_sim,3)


non_sim, sim = cut_by_sim_threshold(0.4,colex_edges_sort_by_sim)

# np.save("colex_edges_sort_by_lan",colex_edges_sort_by_lan,allow_pickle=True)
# np.save("colex_edges_sort_by_sim",colex_edges_sort_by_sim,allow_pickle=True)

# np.save("non_sim_t_40",non_sim,allow_pickle=True)
# np.save("sim_t_40",sim,allow_pickle=True)
# np.save("concept_dict_t",[concept_dict_t],allow_pickle=True)



