import pandas as pd
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv
import pickle

def top_n_argmax(a,n,format):
    id_list = []
    name = []
    count = []
    for i in range(n):
        id = np.argmax(a)
        print(id)
        id_list.append(id)
        name.append((format[id]))
        count.append(a[id])

        a[id] -= 99999

    return id_list,name,count


def randomly_pick_n_samples(pairs,n):
    picked = []
    for l in pairs:
        ids =  np.random.randint(0, high =len(l), size =n)
        a = np.array(l)
        p = a[ids]
        picked.append(p)
    return picked



format = np.load("label_format.npy",allow_pickle=True)
format_l = list(format)
prediction_relation = "nn_data/prediction/final_20k_5_fold_t_2.csv"
exists_relation = "v2.2_remove_redaundent/has_relation/glove_common_crawl_relations_v2.csv"

count1 = []
for r in format:
    count1.append(0)

relation_index1 = []

top_10_conceptnet_id = list(np.load("nn_data/piechart/conceptnet_top10_id.npy",allow_pickle=True))
top_10_conceptnet = np.load("nn_data/piechart/conceptnet_top10_name.npy",allow_pickle=True)
top_10_conceptnet_pairs = []
for i in range(10):
    top_10_conceptnet_pairs.append([])

with open(exists_relation,encoding="utf-8") as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        relation = row[4:]
        # print(row)
        # print(row)
        # print(relation)
        if relation != ['relation']:
            for r in relation:
                index = format_l.index(r)
                if index in top_10_conceptnet_id:
                    id = top_10_conceptnet_id.index(index)
                    top_10_conceptnet_pairs[id].append([row[1],row[2]])
                relation_index1.append(index)
                count1[index] += 1




# plt.hist(relation_index1, bins=bin)  # arguments are passed to np.histogram
# plt.title("Histogram for relations that exists in conceptnet")
# plt.xlabel("relationship index")
# plt.ylabel("# of colexifications")
# plt.savefig("nn_data/histogram_exist")
# plt.show()
# plt.close()




count2 = []
for r in format:
    count2.append(0)


top_10_predict_id = list(np.load("nn_data/piechart/prediction_top10_id_20k.npy",allow_pickle=True))
top_10_predict = np.load("nn_data/piechart/prediction_top10_name_20k.npy",allow_pickle=True)
top_10_predict_pairs = []



for i in range(10):
    top_10_predict_pairs.append([])

relation_index2 = []

with open(prediction_relation,encoding="utf-8") as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        # print(row)
        relation = row[3:]
        # print(row)
        # print(relation)
        if relation != ['relation']:
            for r in relation:
                index = format_l.index(r)
                if index in top_10_predict_id:
                    id = top_10_predict_id.index(index)
                    top_10_predict_pairs[id].append([row[1],row[2]])
                relation_index2.append(index)
                count2[index] += 1



# plt.hist(relation_index2, bins=bin)  # arguments are passed to np.histogram
# plt.title("Histogram for relations predicted by neural net")
# plt.xlabel("relationship index")
# plt.ylabel("# of colexifications")
# plt.savefig("nn_data/histogram_prediction")
# plt.show()
# plt.close()






# Pie chart, where the slices will be ordered and plotted counter-clockwise:

 # only "explode" the 2nd slice (i.e. 'Hogs')

count3 = []
for i in range(len(count2)):
    count3.append(count2[i] +count1[i])

fig5, ax5 = plt.subplots()
ax5.pie(count3, labels=format_l, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax5.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Pie chart for relations both by prediction and conceptnet")
# plt.savefig("nn_data/piechart/exist_predict_all")
plt.show()
plt.close()


top_10_relation_id, top_10_relation,top_10_relation_count = top_n_argmax(count3,10,format)


fig6, ax6 = plt.subplots()
ax6.pie(top_10_relation_count[2:], labels=top_10_relation[2:], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax6.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Pie chart for relations both by prediction and conceptnet")
# plt.savefig("nn_data/piechart/exist_predict_top_10")
plt.show()
plt.close()



fig4, ax4 = plt.subplots()
ax4.pie(count1, labels=format_l, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Pie chart for relations exists in concept net")
# plt.savefig("nn_data/piechart/exist_all")
plt.show()
plt.close()

top_10_relation_exist_id, top_10_relation_exist,top_10_relation_exist_count = top_n_argmax(count1,10,format)
# np.save("nn_data/piechart/conceptnet_top10_id.npy",top_10_relation_exist_id,allow_pickle=True)
# np.save("nn_data/piechart/conceptnet_top10_name.npy",top_10_relation_exist,allow_pickle=True)


fig3, ax3 = plt.subplots()
ax3.pie(top_10_relation_exist_count[2:], labels=top_10_relation_exist[2:], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Pie chart for relations exists in concept net")
# plt.savefig("nn_data/piechart/exist_top_10")
plt.show()
plt.close()

bin = np.arange(0,len(format),1)


fig1, ax1 = plt.subplots()
ax1.pie(count2, labels=format_l, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Pie chart for relations predicted by neural net")
plt.savefig("nn_data/piechart/predict_all_20k")
plt.show()
plt.close()

top_10_predict_id, top_10_predict,top_10_count = top_n_argmax(count2,10,format)

x = randomly_pick_n_samples(top_10_predict_pairs,10)

# np.save("nn_data/piechart/prediction_top10_id.npy",top_10_predict_id,allow_pickle=True)
# np.save("nn_data/piechart/prediction_top10_name.npy",top_10_predict,allow_pickle=True)

fig2, ax2 = plt.subplots()
ax2.pie(top_10_count[2:], labels=top_10_predict[2:], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Pie chart for relations predicted by neural net")
plt.savefig("nn_data/piechart/predict_top_10_20k")
plt.show()
plt.close()

