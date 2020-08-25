
import pickle
from scipy.io import loadmat, savemat
from representations.sequentialembedding import SequentialEmbedding
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import scipy
import pandas as pd
# help fucntions

# dbfile = open("chapter_words_2", 'rb')
# chapter_words = pickle.loads(dbfile)
# dbfile.close()
dont_have = []

def average_chapter_similarity(word, word_chapters, emd, s, n):
    emd_s = emd.embeds[s]
    # emd_e = emd.embeds[e]

    n_s = emd_s.closest(word, n+1)
    # n_e = emd_e.closest(word,101)

    L = []
    if n_s == "we don't have this word":
        print("we don't have    ",word)
        dont_have.append(word)
        L.append(np.zeros(22))
    else:
        for sim,neigh in n_s:
            N = []
            for chapter in word_chapters:
                c = 0
                for w in chapter:
                    # print(w)
                    c += emd_s.similarity( neigh,w )
                c = c / len(chapter)
                N.append(c)
            L.append(N)
    return L



def filter_chapter_list(L):
    R = []
    for chapter in L:
        C = []
        for item in chapter:
            I = item.replace(',','')
            S = I.split()
            C.append(S[0])
        R.append(C)
    return R


def pair_and_sort(fields, sim):
    L = []
    for i in range(22):
        L.append([sim[i],fields[i]])
    return L

def max_one_sim_counts(sim):
    R = np.zeros(22)
    for item in sim:
        id = np.argmax(item)
        R[id] += 1
    return R

def max_3_sim_counts(sim,weight = [1,0.75,0.5]):
    R = np.zeros(22)
    for item in sim:
        copy = np.copy(item)
        id1 = np.argmax(copy)
        # print(copy)
        # print(len(copy))
        # print(id1)
        copy[id1] -= 1000
        id2 = np.argmax(copy)
        copy[id2] -= 1000
        id3 = np.argmax(copy)
        R[id1] += weight[0]
        R[id2] += weight[1]
        R[id3] += weight[2]
    return R

def first_3_arguments(sim):
    R = np.zeros(22)
    for item in sim:
        copy = np.copy(item)
        id1 = np.argmax(copy)
        # print(copy)
        # print(len(copy))
        # print(id1)
        copy[id1] -= 1000
        id2 = np.argmax(copy)
        copy[id2] -= 1000
        id3 = np.argmax(copy)
        R[id1] += weight[0]
        R[id2] += weight[1]
        R[id3] += weight[2]
    return R


def one_word_list(chapters):
    L = []
    for chapter in chapters:
        L.extend(chapter)
    return L

def get_all_max_one_sim_counts(f_list, f_chap_list,embds,year,n = 20):
    R = []
    for i in range(len(f_list)):
        word = f_list[i]
        # print(i, word)
        L = average_chapter_similarity(word,f_chap_list,embds,year,n)
        L2 = max_one_sim_counts(L)
        R.append(L2)
    return R

def get_all_max_3_sim_counts(f_list, f_chap_list,embds,year,weight = [1.,0.75,0.5], n = 20):
    R = []
    for i in range(len(f_list)):
        word = f_list[i]
        # print(i, word)
        L = average_chapter_similarity(word,f_chap_list,embds,year,n)
        L2 = max_3_sim_counts(L,weight)
        R.append(L2)
    return R

def counts_2_best_chapter(sim):
    L = []
    for a in sim:
        copy = np.copy(a)
        id1 = np.argmax(copy)
        copy[id1] -= 1000
        id2 = np.argmax(copy)
        L.append([id1,id2])
    return L

def correct_rates(sim,colex):
    first = 0
    second = 0
    index = []
    for i in range(len(sim)):
        pair1 = sim[i]
        pair2 = colex[i]
        if pair1[0] == pair2[0]:
            first += 1
        if pair1[1] == pair2[1]:
            second +=1
        if pair1[0] == pair2[0] and pair1[1] == pair2[1]:
            index.append(i)

    first = first / 1081.
    second = second / 1081.
    return [first, second] , index

def correct_rates_v2(sim,colex):
    c = 0
    for i in range(len(sim)):
        pair1 = sim[i]
        pair2 = colex[i]
        if pair1[0] in pair2:
            c += 1
        if pair1[1] in pair2:
            c += 1
    c = c /2.
    c = c / 1081.
    return c

def get_n_neighbors_of_word_list( l, embds,f_list,year,n=100):
    r = []
    for word_id in l:
        word = f_list[word_id]
        emd_y = embds.embeds[year]
        r.append(emd_y.closest(word, n + 1))
    return r

def get_average_similarity_of_neighbor_list(neigh,f_list,year,embd):
    embd_y = embd.embeds[year]
    list = []
    for word in f_list:
        c = 0
        for pair in neigh:
            n = pair[1]
            c += embd_y.similarity(word, n)
        c = c / len(neigh)
        list.append(c)
    return list

def get_average_similarity_of_neighbor_list_of_list(n_list,f_list,year,embd):
    R = []
    for l in n_list:
        R.append(get_average_similarity_of_neighbor_list(l,f_list,year,embd))
    return R

def transfer_similarity_list_to_arg_n_max_of_word_list(s,n,f_list):
    R = []
    for sim in s:
        ids = arg_n_max_of_list(sim,n)
        words = transfer_idlist_wordlist(ids,f_list)
        R.append(words)
    return R




def arg_n_max_of_list(l,n):
    list = []
    while len(list) < n:
        id = np.argmax(l)
        list.append(id)
        l[id] = -9999
    return list

def transfer_idlist_wordlist(ids, f_list):
    L = []
    for id in ids:
        L.append(f_list[id])
    return L



dbfile = open('fields.p', 'rb')
fields = pickle.load(dbfile)
dbfile.close()

chapter_words_np = np.load("chapter_list.npy",allow_pickle = True)
colex_words_chapters = np.load("colex_words_chapters_averaged.npy",allow_pickle = False)
# embds = SequentialEmbedding.load("embeddings/eng-all_sgns/sgns", range(1890, 2000, 10))
year_s = 1890
year_end = 1990
target = 'fire'
f_chap_list = filter_chapter_list(chapter_words_np)
f_list = one_word_list(f_chap_list)

l1 = np.load("test_list1.npy",allow_pickle = False)
l2 = np.load("test_list2.npy",allow_pickle = False)



# n1 = get_n_neighbors_of_word_list(l1, embds,f_list,year_s)
# n2 = get_n_neighbors_of_word_list(l2, embds,f_list,year_end)
n1 = np.load("test_neighbor1.npy",allow_pickle = False)
n2 = np.load("test_neighbor2.npy",allow_pickle = False)

# s1 = get_average_similarity_of_neighbor_list_of_list(n1,f_list,year_s,embds)
# s2 = get_average_similarity_of_neighbor_list_of_list(n2,f_list,year_end,embds)
s1 = np.load("test_average_sim1.npy",allow_pickle = False)
s2 = np.load("test_average_sim2.npy",allow_pickle = False)

w1 = transfer_similarity_list_to_arg_n_max_of_word_list(s1,20,f_list)
w2 = transfer_similarity_list_to_arg_n_max_of_word_list(s2,20,f_list)

# test
# L = get_average_similarity_of_neighbor_list(n1[0],f_list,year_s,embds)
# max_id = arg_n_max_of_list(L,20)
# max_words = transfer_idlist_wordlist(max_id , f_list)
# np.save("test_neighbor2",n1,allow_pickle=False)

#--------------------------calculatting rate ------------------------#

# max_one_sim_1890 = get_all_max_one_sim_counts(f_list, f_chap_list,embds,year_s,n = 100)
# max_one_sim_1990 = get_all_max_one_sim_counts(f_list, f_chap_list,embds,year_end, n = 100)
#
# np.save("max_one_sim_1890",max_one_sim_1890,allow_pickle=True)
# np.save("max_one_sim_1990",max_one_sim_1990,allow_pickle=True)

#---------------------------------calculateing rate------------------#
# max_one_sim_1990 = np.load("max_one_sim_1990.npy",allow_pickle = True)
# max_one_sim_1890 = np.load("max_one_sim_1890.npy",allow_pickle = True)
#
# count_1990 = counts_2_best_chapter(max_one_sim_1990)
# count_1890 = counts_2_best_chapter(max_one_sim_1890)
# count_colex = counts_2_best_chapter(colex_words_chapters)
#
#
# rate1 = correct_rates(count_colex,count_1990)
# rate2 = correct_rates(count_colex,count_1890)
# print("rate1: ",rate1) # [0.3755781683626272, 0.13506012950971322] 1990
# print("rate2: ",rate2)  # [0.4061054579093432, 0.1341350601295097] 1890
#--------------------------calculatting rate ------------------------#

# R = get_all_max_one_sim_counts(f_list, f_chap_list,embds,year_end)

# w1 = [1.0.75,0.5]
# print("start")
# w1_1890 = get_all_max_3_sim_counts(f_list, f_chap_list,embds,year_s,weight = [1.,0.75,0.5])
# w1_1990 = get_all_max_3_sim_counts(f_list, f_chap_list,embds,year_end,weight = [1.,0.75,0.5])
# print("1")
# w2 = [1,1,0.5]
# w2_1890 = get_all_max_3_sim_counts(f_list, f_chap_list,embds,year_s,weight = [1.,1,0.5])
# print("2")
# w2_1990 = get_all_max_3_sim_counts(f_list, f_chap_list,embds,year_end,weight = [1.,1,0.5])
# print("3")
# print("end")
# L = average_chapter_similarity(target, f_chap_list, embds, year_s)
# L2 = max_one_sim_counts(L)
# word = 'fire'
#
# emd_s = embds.embeds[year_s]
# emd_e = embds.embeds[year_end]
#
# t1 = emd_s.closest(word, 101)
# t2 = emd_e.closest(word, 101)

# Rremove = 'dandruff' headbandcollarbone

# dont_have = ["sibling"]
#
# for word in :
#     emd_s.closest()

# np.save("max_one_sim_1890",R,allow_pickle=True)
# np.save("max_3_sim_1890",w1_1890,allow_pickle=True)
# np.save("dont_have",dont_have,allow_pickle=True)

# max_3_sim_1890_w1 = np.load("max_3_sim_1890.npy",allow_pickle = True)
# count_max3_1890_w1 = counts_2_best_chapter(max_3_sim_1890_w1)
# rate3 = correct_rates(count_colex,count_max3_1890_w1)  # [0.42738205365402404, 0.13506012950971322]
# np.save("max_3_sim_1990",w1_1990 ,allow_pickle=True)
# np.save("max_3_sim_1890_w2",w2_1890,allow_pickle=True)
# np.save("max_3_sim_1990_w2",w2_1990,allow_pickle=True)

# count_max3_1990_w1 = counts_2_best_chapter(w1_1990)
# count_max3_1890_w2 = counts_2_best_chapter(w2_1890)
# count_max3_1990_w2 = counts_2_best_chapter(w2_1990)

# rate4 = correct_rates(count_colex,count_max3_1990_w1) # [0.38575393154486587, 0.11933395004625347]
# rate5 = correct_rates(count_colex,count_max3_1890_w2) # [0.4283071230342276, 0.13135985198889916]
# rate6 = correct_rates(count_colex,count_max3_1990_w2) # [0.39037927844588344, 0.1211840888066605]
#n = 20
##('rate1: ', [0.4070305272895467, 0.12303422756706753])
##('rate2: ', [0.4366327474560592, 0.1091581868640148])
#n = 10
#('rate1: ', [0.4107308048103608, 0.11655874190564293])
#('rate2: ', [0.4357076780758557, 0.11100832562442182])
#n = 50
#('rate1: ', [0.38390379278445885, 0.12950971322849214])
#('rate2: ', [0.42645698427382056, 0.12488436632747456])

# ('rate1: ', 0.3834412580943571)
# ('rate2: ', 0.41211840888066603)
