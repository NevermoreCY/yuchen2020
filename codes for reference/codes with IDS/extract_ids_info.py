from scipy.io import loadmat, savemat
import pandas as pd
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

M1 = loadmat('data/colexMats.mat')
M2 = loadmat('data/MCColex.mat')
languagefamily = pd.read_csv('data/languagefamily.csv')
dictionary = pd.read_csv('data/dictionary.csv')
macroarea = pd.read_csv('data/geography.csv')
usf_matrices = loadmat('data/usf_colexMats.mat')
hbc_matrices = loadmat('data/hbc_colexMats.mat')

chapters = [0,65,104,186,309,379,432,466,515,575,644,686,739,766,799,844,891,929,969,1001,1035,1061,1081]


def get_chapter_list(chap, dict):
    L = []
    for i in range(len(chap)-1):
        list = dict[chap[i] : chap[i+1]]
        L.append(list)
    return L

chapter_list = get_chapter_list(chapters, dictionary["ids_name"])


def get_colexifications_of_each_words_each_chapter(chap,M):
    rate_of_colex = []
    for i in range(chap[-1]):
        print(i)
        list = np.zeros(22)
        for key in M:
            if key != "__header__" and key != "__version__" and key != "__globals__":
                A = M[key].toarray()
                for j in range(22):
                    c = A [i, chap[j]:chap[j+1]].sum()
                    c = c / (chap[j+1] - chap[j])
                    list[j] += c
        rate_of_colex.append(list)
    return rate_of_colex

def get_colex_rate_all_domain(chap, M):
    rate_of_colex = []
    for i in range(len(chap)-1):
        s = chap[i]
        e = chap[i+1]
        count = 0
        for key in M:
            if key != "__header__" and key != "__version__" and key != "__globals__":
                A = M[key].toarray()
                c1 = A[s:e,s:e].sum()
                c2 = A[s:e,:s].sum() + A[s:e,e:].sum()
                count = count + c1 /2 + c2
        rate_of_colex.append(count / (e-s) / (len(M)-3))
    return rate_of_colex

def filter_the_M(M,zero):
    new_dic = {}
    for key in M:
        if key != "__header__" and key != "__version__" and key != "__globals__":
            A = M[key].toarray()
            for i in zero:
                A[i,:] = 0
                A[:,i] = 0
            new_dic[key] = A
    return new_dic


def counter_index(nd,length):
    r = []
    for i in range(length):
        if i not in nd:
            r.append(i)
    return r


def get_noun_colex_rate_all_domain(chap, M):
    rate_of_colex = []
    for i in range(len(chap)-1):
        s = chap[i]
        e = chap[i+1]
        count = 0
        for key in M:
                A = M[key]
                c1 = A[s:e,s:e].sum()
                c2 = A[s:e,:s].sum() + A[s:e,e:].sum()
                count = count + c1 /2 + c2
        rate_of_colex.append(count / (e-s) / (len(M)-3))
    return rate_of_colex


def counts_2_best_chapter(sim):
    L = []
    for a in sim:
        copy = np.copy(a)
        id1 = np.argmax(copy)
        copy[id1] -= 1000
        id2 = np.argmax(copy)
        L.append([id1,id2])
    return L

def guess_self_chap_rate(chap,colex):
    correct = 0
    for i in range(len(chap)-1):
        predict = i
        for j in range(chap[i],chap[i+1]):
            most = np.argmax(colex[j])
            if most == predict:
                correct += 1.
    correct = correct / 1081
    return correct



def get_colexifications_of_each_words_each_chapter(chap,M):
    rate_of_colex = []
    for i in range(chap[-1]):
        print(i)
        list = np.zeros(22)
        for key in M:
            if key != "__header__" and key != "__version__" and key != "__globals__":
                A = M[key].toarray()
                for j in range(22):
                    c = A [i, chap[j]:chap[j+1]].sum()
                    c = c / (chap[j+1] - chap[j])
                    list[j] += c
        rate_of_colex.append(list)
    return rate_of_colex

def colexification_of_word(word_id,M):
    c = np.zeros(1081)
    for key in M:
        if key != "__header__" and key != "__version__" and key != "__globals__":
            A = M[key].toarray()
            c += A[word_id,:]
    return c

def colexification_of_word_list(L,M):
    l = []
    for word_id in L:
        print(word_id)
        l.append(colexification_of_word(word_id,M))
    return l

def find_colex_words_of_word_counts(C,D):
    r = []
    for c in C:
        words = []
        for i in range(len(c)):
            if c[i] > 0:
                a = [c[i], D[i]]
                words.append(a)
        r.append(words)
    return r


def find_overlap_with_counts(n,w):
    overlap =[]
    for i in range(len(n)):
        neigh = n1[i][:,1]
        colex = np.array(w[i])
        c_list = colex[:,1]
        o = []
        for i in range(len(c_list)):
            if c_list[i] in neigh:
                o.append(list(colex[i]))
        overlap.append(o)
    return overlap

def check_high_counted_overlap_ratio( w,over,c = 2):
    total = 0.
    overlap = 0.
    for i in range(len(w)):
        w_list = w[i]
        over_list = over[i]
        for item in w_list:
            count = item[0]
            cc = [str(count),item[1]]
            if count >= c:
                total += 1.
                if cc in over_list:
                    overlap += 1.
    print(overlap / total)
    return overlap / total















#---------------------p value section------------------------------
# # R = get_colex_rate_all_domain(chapters, M1)
#
# dbfile = open("noun_index", 'rb')
# noun_index = pickle.load(dbfile)
# dbfile.close()
#
# dbfile = open("adj_index", 'rb')
# adj_index = pickle.load(dbfile)
# dbfile.close()
#
# dbfile = open("verb_index", 'rb')
# verb_index = pickle.load(dbfile)
# dbfile.close()
#
# # remove duplicate
# noun_index = list(dict.fromkeys(noun_index))
# adj_index = list(dict.fromkeys(adj_index))
# verb_index = list(dict.fromkeys(verb_index))
#
# non_noun_index = counter_index(noun_index,1081)
# non_adj_index = counter_index(adj_index,1081)
# non_verb_index = counter_index(verb_index,1081)
#
# M_noun = filter_the_M(M1,non_noun_index)
# M_adj = filter_the_M(M1,non_adj_index)
# M_verb = filter_the_M(M1,non_verb_index)
#
# #
# R_noun = get_noun_colex_rate_all_domain(chapters, M_noun)
# R_adj = get_noun_colex_rate_all_domain(chapters, M_adj)
# R_verb = get_noun_colex_rate_all_domain(chapters, M_verb)
#---------------------p value section------------------------------


# R = get_colexifications_of_each_words_each_chapter(chapters,M1)
# dbfile = open("colex_words_chapters", 'rb')
# colex_word_chap = pickle.load(dbfile)
# dbfile.close()



# dbfile = open("chapter_words", 'wb')
# pickle.dump(chapter_list,dbfile)
# dbfile.close()
# dbfile = open("chapter_words_2", 'wb')
# pickle.dumps(chapter_list, protocol=0, 'latin-1')
# dbfile.close()
#
# np.save("chapter_words_np",chapter_list,allow_pickle=True)
# np.save("colex_words_chapters",colex_word_chap,allow_pickle=False)

#-----------------------Try to find a new way out--------------------------------------
l1 = np.load("test_list1.npy",allow_pickle = False)
l2 = np.load("test_list2.npy",allow_pickle = False)


colex_average = np.load("colex_words_chapters_averaged.npy",allow_pickle = False)

rate = guess_self_chap_rate(chapters,colex_average)

# c1 = colexification_of_word_list(l1,M1)
# c2 = colexification_of_word_list(l2,M1)
c1 = np.load("test_count1.npy",allow_pickle = False)
c2 = np.load("test_count2.npy",allow_pickle = False)

n1 = np.load("test_neighbor1.npy",allow_pickle = False)
n2 = np.load("test_neighbor2.npy",allow_pickle = False)

w1_20 = np.load("test_20_most_sim_neigh_1.npy",allow_pickle = False)
w2_20 = np.load("test_20_most_sim_neigh_2.npy",allow_pickle = False)

f_list = np.load("f_list.npy",allow_pickle = False)


w1 = find_colex_words_of_word_counts(c1 ,f_list )
w2 = find_colex_words_of_word_counts(c2 ,f_list )

over1 = find_overlap_with_counts(n1,w1)
over2 = find_overlap_with_counts(n2,w2)


r1_1 = check_high_counted_overlap_ratio( w1,over1,c = 1)
r2_1= check_high_counted_overlap_ratio( w2,over2,c = 1)

r1_2 = check_high_counted_overlap_ratio( w1,over1,c = 2)
r2_2= check_high_counted_overlap_ratio( w2,over2,c = 2)


r1_3 = check_high_counted_overlap_ratio( w1,over1,c = 3)
r2_3= check_high_counted_overlap_ratio( w2,over2,c = 3)

r1_4 = check_high_counted_overlap_ratio( w1,over1,c = 4)
r2_4= check_high_counted_overlap_ratio( w2,over2,c = 4)

rate1890 = [r1_1,r1_2,r1_3,r1_4]
rate1990 = [r2_1,r2_2,r2_3,r2_4]
x = [1,2,3,4]

plt.plot(x,rate1890,label = "overlap_ratio_1890")
plt.plot(x,rate1990,label = "overlap_ratio_1990")
plt.xlabel("lower bound of number of colexification")
plt.ylabel("overlap ratio")
plt.title('overlap ratio with different lower bound')
plt.legend()
# plt.savefig('overlap_ratio.png')
plt.show()
plt.close()

