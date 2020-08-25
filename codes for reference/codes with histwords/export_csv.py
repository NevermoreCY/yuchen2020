import csv
import numpy as np

def creat_row_list(dict, n = 50):
    count = 0
    row_list = [["concepts", "colexifications"]]
    for key in dict:
        row = [key]
        row.extend(dict[key])
        row_list.append(row)
        count += 1
        if count == n:
            return row_list
    return row_list

def reverse_and_add_first_line(a,format):
    l = list(a)
    l.append(format)
    l.reverse()
    return np.array(l)

#
# clics_1_dict = np.load("clics_1_dict.npy",allow_pickle = True)
# clics_1_dict = clics_1_dict[0]
# clics_2_dict = np.load("clics_2_dict.npy",allow_pickle = True)
# clics_2_dict = clics_2_dict[0]
# c2_row_list = creat_row_list(clics_2_dict, n = 50)
# c1_row_list = creat_row_list(clics_1_dict, n = 50)
#
#
#
# with open('clics_2_colex.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(c2_row_list)
#
# with open('clics_1_colex.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(c1_row_list)

#--------------------7.23.2020---------------------------------------------

# np.save("non_sim_t_40",non_sim,allow_pickle=True)
# np.save("sim_t_40",sim,allow_pickle=True)
format = ['concept 1','concept 2','language weight', "similarity"]

colex_edges_sort_by_lan = np.load("colex_edges_sort_by_lan.npy",allow_pickle = True)
colex_edges_sort_by_lan = reverse_and_add_first_line(colex_edges_sort_by_lan,format)

# colex_edges_sort_by_lan = np.append(colex_edges_sort_by_lan,format)
# colex_edges_sort_by_lan = np.flip(colex_edges_sort_by_lan)



colex_edges_sort_by_sim = np.load("colex_edges_sort_by_sim.npy",allow_pickle = True)
colex_edges_sort_by_sim = reverse_and_add_first_line(colex_edges_sort_by_sim,format)
non_sim_t_40 = np.load("non_sim_t_40.npy",allow_pickle = True)
non_sim_t_40 = reverse_and_add_first_line(non_sim_t_40 ,format)
sim_t_40 = np.load("sim_t_40.npy",allow_pickle = True)
sim_t_40 = reverse_and_add_first_line(sim_t_40,format)

with open('colex_edges_sort_by_lan.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(colex_edges_sort_by_lan)

with open('colex_edges_sort_by_sim.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(colex_edges_sort_by_sim)

with open('non_sim_t_40.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(non_sim_t_40)

with open('sim_t_40.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(sim_t_40)
