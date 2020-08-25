
import pickle
# from representations.sequentialembedding import SequentialEmbedding
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import scipy
import pandas as pd
# help fucntions

def extract_words(L):
    R =[]
    for item in L:
        s = item.split()
        if 'the' in s:
            s.remove('the')
        if len(s) == 1 or "or" in s:
            R.append(s[0])
    return R

def filter_dic(dic, field):
    filtered_dic = {}
    for f in field:
        words = dic[f]
        filter_words = extract_words(words)
        filtered_dic[f] = filter_words
    return filtered_dic

def extract_verbs_adj(L):
    R = []
    for item in L:
        I = item.replace('(1)','')
        I = I.replace('(2)','')
        I = I.replace('(3)', '')
        I = I.replace('/', ' nb ')
        I = I.replace('-', '')
        I = I.replace(' or ', ' ')
        S = I.split()
        if 'to' in S:
            S.remove('to')
        if len(S) == 1 or "nb" in S:
            R.append(S[0])
    return R


def filter_dic_verbs_adj(dic, field):
    filtered_dic = {}
    for f in field:
        words = dic[f]
        filter_words = extract_verbs_adj(words)
        filtered_dic[f] = filter_words
    return filtered_dic

def degree_of_semantic_change_dic(dic,fields,year_s,year_end,embd):
    degree_dic = {}
    for f in fields:
        degree_dic[f] = degree_of_semantic_change(year_s, year_end, dic[f], embd)
    return degree_dic


def degree_of_semantic_change(year_s,year_end,word_list,embd):
    R = []

    emd_s = embd.embeds[year_s]
    emd_e = embd.embeds[year_end]
    for word in word_list:
        # print "Doing",word
        #
        t1 = emd_s.closest(word, 101)
        t2 = emd_e.closest(word, 101)

        c = -1
        for score, neigh in t1:
            for score2, neigh2 in t2:
                if neigh == neigh2:
                    c += 1
        R.append(c)
    return R

def check_dic(dic,fields,year_s,year_end,embd):
    R = []
    for f in fields:
        new_r = check_all_words(year_s, year_end, dic[f], embd)
        R.extend(new_r)
    return R


def check_all_words(year_s,year_end,word_list,embd):
    R= []
    emd_s = embd.embeds[year_s]
    emd_e = embd.embeds[year_end]
    for word in word_list:

        #
        t1 = emd_s.closest(word, 101)
        t2 = emd_e.closest(word, 101)

        if t1 == "we don't have this word" or t2 == "we don't have this word":
            R.append(word)

    return R

def remove_words_from_dic(dic, words, fields):
    for word in words:
        for f in fields:
            if word in dic[f]:
                dic[f].remove(word)


def get_mean_paris(number_dic,fields):
    pairs = []
    for i in range(22):
        f = fields[i]
        if number_dic[f] == []:
            pairs.append((0,f))
            # pairs.append((0, f))
        else:
            x = 1 - (np.array(number_dic[f]) / 100.)
            m = np.mean(x)
            pairs.append((m,f))
    return pairs

def size_of_dic(dic):
    size = 0
    for key in dic:
        l = dic[key]
        size += len(l)
    return size


# get all fields

dbfile = open('fields.p', 'rb') 
fields = pickle.load(dbfile)
dbfile.close()
#write file in pickle
#dbfile = open("fields", 'wb')
#pickle.dump(fields,dbfile)
#dbfile.close()

dbfile = open('verb.p', 'rb') 
dic_verb = pickle.load(dbfile)
dbfile.close()

dbfile = open('noun.p', 'rb') 
dic_noun = pickle.load(dbfile)
dbfile.close()

dbfile = open('adjective.p', 'rb') 
dic_adj = pickle.load(dbfile)
dbfile.close()


# filtered_verb = filter_dic_verbs_adj(dic_verb,fields)
# filtered_adj = filter_dic_verbs_adj(dic_adj,fields)

# second processing for words 
# tpw = dic_noun['The physical world']
# tpw = extract_words(tpw)
#
# filtered_dic = filter_dic(dic_noun, fields)
dbfile = open("filtered_dic_remove_words", 'rb')
filtered_dic = pickle.load(dbfile)
dbfile.close()

dbfile = open("filtered_verb", 'rb')
filtered_verb = pickle.load(dbfile)
dbfile.close()

dbfile = open("filtered_adj", 'rb')
filtered_adj = pickle.load(dbfile)
dbfile.close()


#
# embds = SequentialEmbedding.load("embeddings/eng-all_sgns/sgns", range(1890, 2000, 10))
#
#
# emd_s = embds.embeds[1890]
# emd_e = embds.embeds[1990]
#
# t1 = emd_s.closest('mother-in-law', 101)
# t2 = emd_e.closest('jadeite', 101)
#
# tpw_degree = degree_of_semantic_change(1890,1990,tpw,embds)

# remove words that are not available on HISTRO embeddings
# need_to_remove = check_dic(filtered_dic,fields,1890,1990,embds)
# need_to_remove_verb = check_dic(filtered_verb,fields,1890,1990,embds)
# need_to_remove_adj = check_dic(filtered_adj,fields,1890,1990,embds)
#
# remove_words_from_dic(filtered_dic, need_to_remove, fields)
# remove_words_from_dic(filtered_verb, need_to_remove_verb, fields)
# remove_words_from_dic(filtered_adj, need_to_remove_adj, fields)

# dbfile = open("degree_dic_verb", 'wb')
# pickle.dump(degree_dic_verb,dbfile)
# dbfile.close()
#
# dbfile = open("degree_dic_adj", 'wb')
# pickle.dump(degree_dic_adj,dbfile)
# dbfile.close()


# degree_dic = degree_of_semantic_change_dic(filtered_dic,fields,1890,1990,embds)
# degree_dic_verb = degree_of_semantic_change_dic(filtered_verb,fields,1890,1990,embds)
# degree_dic_adj = degree_of_semantic_change_dic(filtered_adj,fields,1890,1990,embds)
#number dic is the dictionary for number of shared neighbours for each words
# corresopnding to the word in filteed_dic
dbfile = open("degree_dic", 'rb')
number_dic = pickle.load(dbfile)
dbfile.close()
size_noun = size_of_dic(number_dic)


dbfile = open("degree_dic_verb", 'rb')
degree_dic_verb = pickle.load(dbfile)
dbfile.close()
size_verb = size_of_dic(degree_dic_verb)

dbfile = open("degree_dic_adj", 'rb')
degree_dic_adj = pickle.load(dbfile)
dbfile.close()
size_adj = size_of_dic(degree_dic_adj)

# # for f in fields:
# kde = KernelDensity(bandwidth=1.0, kernel='gaussian',metric='euclidean')
# x = np.array([30,20,30,30,20,20,20,30,30,20,20])
# # x = x / x.sum()
# kde.fit(x[:,None])
# x_d = np.linspace(0, 100, 100)
# logprob = kde.score_samples(x_d[:, None])
# y = np.exp(logprob)
# plt.plot(x_d,y)
# # plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
# plt.show()
# plt.close()
# kernel density estimation--------------621 ceiling
# plt.figure(figsize=(20,20))
# for f in fields:
#     if f != 'Miscellaneous function words':
#         print(f)
#         bw = 0.05
#         x = 1 - (np.array(number_dic[f]) / 100.)
#         kde = KernelDensity(bandwidth= bw, kernel='gaussian')
#         kde.fit(x[:, None])
#         x_d = np.linspace(0, 1.0,100)
#         logprob = kde.score_samples(x_d[:, None])
#         y = np.exp(logprob)
#         print(y.sum())
#         y = y * 0.05
#         plt.plot(x_d, y, label = f)
#         # plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
#
# plt.xlabel("degree of semantic change")
# plt.ylabel("probability density")
# plt.title('juxtapose Kde plot with Bandwidth = ' + str(bw))
# plt.legend()
# # plt.savefig('juxtapose_kde.png')
# plt.show()
# plt.close()

# HISTO PLOTS------------------------------------621 floor
# for f in fields:
#     degree = 1 - (np.array(degree_dic[f]) / 100.)
#     plt.hist(degree, bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
#     plt.xlabel("degree of semantic change")
#     plt.ylabel("number of samples")
#     plt.title('histogram for field '+f)
#     plt.savefig('histo/'+f+'.png')
#     plt.show()
#     plt.close()

# dbfile = open("degree_dic", 'wb')
# pickle.dump(degree_dic,dbfile)
# dbfile.close()
#
# dbfile = open("filtered_dic", 'wb')
# pickle.dump(filtered_dic,dbfile)
# dbfile.close()

# dbfile = open("removed_words_1", 'wb')
# pickle.dump(need_to_remove,dbfile)
# dbfile.close()

# dbfile = open("filtered_dic_remove_words", 'wb')
# pickle.dump(filtered_dic,dbfile)
# dbfile.close()


# tpw_d = degree_dic['The physical world']
# tpw_w = filtered_dic['The physical world']
#
# set = [[],[],[],[],[],[],[],[],[],[]]
# for i in range(len(tpw_d)):
#     if 0 <= tpw_d[i] < 10:
#         set[0].append(tpw_w[i])
#     if 10 <= tpw_d[i] < 20:
#         set[1].append(tpw_w[i])
#     if 20 <= tpw_d[i] < 30:
#         set[2].append(tpw_w[i])
#     if 30 <= tpw_d[i] < 40:
#         set[3].append(tpw_w[i])
#     if 40 <= tpw_d[i] < 50:
#         set[4].append(tpw_w[i])
#     if 50 <= tpw_d[i] < 60:
#         set[5].append(tpw_w[i])
#     if 60 <= tpw_d[i] < 70:
#         set[6].append(tpw_w[i])
#     if 70 <= tpw_d[i] < 80:
#         set[7].append(tpw_w[i])
#     if 80 <= tpw_d[i] < 90:
#         set[8].append(tpw_w[i])
#     if 90 <= tpw_d[i] <= 100:
#         set[9].append(tpw_w[i])


colex_rate = [61.46923076923077,
 66.15384615384616,
 23.829268292682926,
 40.46747967479675,
 46.714285714285715,
 37.216981132075475,
 48.088235294117645,
 50.19387755102041,
 55.94166666666667,
 57.028985507246375,
 73.79761904761905,
 59.5188679245283,
 24.35185185185185,
 51.40909090909091,
 63.422222222222224,
 45.5531914893617,
 64.59210526315789,
 55.3625,
 47.796875,
 34.64705882352941,
 23.192307692307693,
 31.925]

colex_rate_per_langeuage_per_word =[0.22031982354563,
 0.23711055969120487,
 0.08540956377305708,
 0.14504473001719265,
 0.1674347158218126,
 0.1333941976060053,
 0.17235926628716003,
 0.17990637115061078,
 0.2005077658303465,
 0.20440496597579347,
 0.26450759515275646,
 0.2133292757151552,
 0.08728262312491702,
 0.18426197458455523,
 0.22731979291119075,
 0.16327308777548996,
 0.23151292209017166,
 0.19843189964157706,
 0.17131496415770608,
 0.12418300653594772,
 0.08312655086848636,
 0.11442652329749105]

R_noun = [0.17555741360089186,
 0.11403753251579339,
 0.04926652527394839,
 0.08592553316837516,
 0.07789855072463768,
 0.08169264424391577,
 0.13554987212276215,
 0.13827270038450162,
 0.040126811594202894,
 0.037859693341734925,
 0.020703933747412008,
 0.0514766201804758,
 0.008252818035426731,
 0.08031400966183576,
 0.017351046698872784,
 0.03330249768732655,
 0.02669717772692601,
 0.05634057971014493,
 0.08967391304347826,
 0.04848678601875533,
 0.023759754738015608,
 0.06721014492753623]

# [0.1810758082497213,
#  0.13219992567818656,
#  0.05423736302580417,
#  0.09789972899728996,
#  0.0869047619047619,
#  0.09095570139458573,
#  0.1517476555839727,
#  0.1414522330671399,
#  0.04933574879227053,
#  0.050278302877546734,
#  0.03183229813664597,
#  0.0589964451736396,
#  0.013150831991411703,
#  0.08816425120772946,
#  0.0249597423510467,
#  0.047872340425531915,
#  0.055587337909992374,
#  0.07110507246376811,
#  0.10495923913043478,
#  0.06745524296675191,
#  0.031424191750278704,
#  0.07083333333333333]

R_adj =[0.006354515050167224,
 0.020624303232998888,
 0.009875397667020148,
 0.005405325792388359,
 0.010119047619047618,
 0.0,
 0.0,
 0.0,
 0.000392512077294686,
 0.0,
 0.002846790890269151,
 0.036710418375717806,
 0.0030193236714975845,
 0.00922266139657444,
 0.08413848631239935,
 0.02913968547641073,
 0.035707475209763535,
 0.0,
 0.0,
 0.0,
 0.000975473801560758,
 0.003170289855072464]
 #    [0.015802675585284278,
 # 0.024897807506503156,
 # 0.009985860728172499,
 # 0.005861906445151408,
 # 0.010196687370600414,
 # 6.836204539239814e-05,
 # 0.00031969309462915604,
 # 0.0003697131026323573,
 # 0.000785024154589372,
 # 0.0004725897920604915,
 # 0.00392512077294686,
 # 0.03732567678424938,
 # 0.0036231884057971015,
 # 0.01290074659639877,
 # 0.09041867954911434,
 # 0.030334566759173606,
 # 0.04581426392067125,
 # 0.00040760869565217395,
 # 0.00011322463768115942,
 # 0.00021312872975277067,
 # 0.001254180602006689,
 # 0.003532608695652174]

R_verb =[0.014130434782608696,
 0.0026941657376439985,
 0.0007069635913750442,
 0.014050901378579003,
 0.02324016563146998,
 0.019517363959529667,
 0.016304347826086956,
 0.024918663117420883,
 0.09725241545893719,
 0.13500315059861373,
 0.19250345065562458,
 0.060363686081487566,
 0.01442565754159957,
 0.022342995169082128,
 0.07008856682769726,
 0.027983348751156337,
 0.12128146453089245,
 0.09044384057971014,
 0.038326539855072464,
 0.04672847399829497,
 0.019439799331103676,
 0.020833333333333332]

 #    [0.03076923076923077,
 # 0.01119472315124489,
 # 0.004727819017320608,
 # 0.025170849534582304,
 # 0.02629399585921325,
 # 0.031856713152857534,
 # 0.021898976982097185,
 # 0.029798876072168,
 # 0.10818236714975846,
 # 0.14083175803402648,
 # 0.19923222912353347,
 # 0.06391851244189226,
 # 0.015968867418142782,
 # 0.02486824769433465,
 # 0.07202093397745571,
 # 0.03866019118100524,
 # 0.1282894736842105,
 # 0.09904891304347825,
 # 0.059895833333333336,
 # 0.054667519181585675,
 # 0.019857859531772576,
 # 0.025452898550724638]

colex_pair = []
for i in range(len(colex_rate_per_langeuage_per_word)):
    colex_pair.append((colex_rate_per_langeuage_per_word[i], fields[i]))

colex_pair_s = colex_pair[:]
colex_pair_s.sort()

mean_pair = get_mean_paris(number_dic,fields)
mean_pair_adj = get_mean_paris(degree_dic_adj,fields)
mean_pair_verb = get_mean_paris(degree_dic_verb,fields)

mean_pair_s = mean_pair[:]
mean_pair_s.sort()
means = []
for item in mean_pair:
    means.append(item[0])

means_adj = []
for item in mean_pair_adj:
    means_adj.append(item[0])

means_verb = []
for item in mean_pair_verb:
    means_verb.append(item[0])


x_axis = range(22)


#normalize two
crplpw_arr = np.array(colex_rate_per_langeuage_per_word)
mean_arr = np.array(means)
d = np.mean(mean_arr) - np.mean(crplpw_arr)
d_n = np.mean(mean_arr) - np.mean(np.array(R_noun))
mean_arr_n = mean_arr - d_n
mean_arr = mean_arr - d


#-------------6 21
# plt.plot(x_axis,crplpw_arr, 'x', label = "colex_rate_per_language_per_word")
# plt.plot(x_axis,means, 'o', label = "mean of degree of semantic change")
# plt.ylabel("degree of semantic change / colexification count per word")
# plt.xlabel("chapters")
# plt.title('degree of semantic change and colexification count per word for 22 chapters')
# plt.legend()
# plt.savefig('Plots/colex&semantic degree.png')
# plt.show()
# plt.close()
#
#---------------------p value section------------------------------
# plt.plot(x_axis,crplpw_arr, 'o', color = 'b')
# plt.plot(x_axis,mean_arr, 'o',color = 'b')
# plt.ylabel("degree of semantic change / colexification count per word")
# plt.xlabel("chapters")
# plt.title('degree of semantic change and colexification count per word for 22 chapters')
#
# plt.savefig('Plots/colex&semantic degree_normed.png')
# plt.show()
# plt.close()
#
# plt.plot(x_axis,R_noun, 'o', color = 'b')
# plt.plot(x_axis,mean_arr_n, 'o',color = 'b')
# plt.ylabel("degree of semantic change / colexification count per word")
# plt.xlabel("chapters")
# plt.title('degree of semantic change and colexification count per word for 22 chapters')
# plt.savefig('Plots/colex&semantic_degree_normed_noun.png')
# plt.show()
# plt.close()
#---------------------p value section------------------------------
#---------------- 6 2 1

#--------------------find noun_index
#---------------------p value section------------------------------
# df = pd.read_excel ('IDS/dictionary.xlsx')
#
# col = df.columns
#
# index = df['index']
# ids_index= df['ids_index']
# ids_name = df['ids_name']
#
#
# noun_index = []
# adj_index = []
# verb_index = []
#---------------------p value section------------------------------
# for key in filtered_dic:
#     print ("key is",key)
#     word_list = filtered_dic[key]
#     for word in word_list:
#         for i in range(len(ids_name)):
#             s = ids_name[i].replace(",", " ")
#             s = s.split()
#             if word in s:
#                 print(word, s)
#                 noun_index.append(i)
# #
# for key in filtered_verb:
#     print ("key is",key)
#     word_list = filtered_verb[key]
#     for word in word_list:
#         for i in range(len(ids_name)):
#             s = ids_name[i].replace(",", " ")
#             s = s.split()
#             if word in s:
#                 print(word,s)
#                 verb_index.append(i)
# #
# for key in filtered_adj:
#     print ("key is",key)
#     word_list = filtered_adj[key]
#     for word in word_list:
#         for i in range(len(ids_name)):
#             s = ids_name[i].replace(","," ")
#             s = s.split()
#             if word in s:
#                 print(word,s)
#                 adj_index.append(i)
#

#------------------------------------
#
# dbfile = open("noun_index", 'wb')
# pickle.dump(noun_index,dbfile)
# dbfile.close()
#
# dbfile = open("verb_index", 'wb')
# pickle.dump(verb_index,dbfile)
# dbfile.close()
#
# dbfile = open("adj_index", 'wb')
# pickle.dump(adj_index,dbfile)
# dbfile.close()

#--------------------find noun_index----------------
#---------------------p value section------------------------------
# total_size = size_adj + size_noun +size_verb
# means_all = (  np.array(means)*size_noun + np.array(means_adj)*size_adj + np.array(means_verb)*size_verb  ) / total_size
# total_size = 641+109+290
# R_all = (np.array(R_noun) * 641 + np.array(R_adj) * 109 + np.array(R_verb)*290) / total_size
#
# c,p = scipy.stats.pearsonr(R_all, means_all )
# c_noun, p_noun = scipy.stats.pearsonr(R_noun, means )
# c_adj,p_adj = scipy.stats.pearsonr(R_adj, means_adj )
# c_verb,p_verb = scipy.stats.pearsonr(R_adj, means_verb )
#
# print(c,p )
# print(c_noun, p_noun )
# print(c_adj,p_adj )
# print(c_verb,p_verb )
#---------------------p value section------------------------------
#--------------------------------end of p_value and P_coefficient

#---------------------------new section------------------------------


