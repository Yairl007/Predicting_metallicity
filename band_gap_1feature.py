import torch
import numpy as np
import pickle as pkl
from pymatgen.ext.matproj import MPRester
import band_gap_classifier_1feature as bgc1
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pymatgen.analysis.local_env as env
import pymatgen.core.structure as st
with MPRester("sLhOEz6qQG8r4Tym") as m:

    def plot_cost(axes, fig, train_cost_list, val_cost_list):
        axes.cla()
        axes.plot(train_cost_list, color='darkslategrey', label='Training')
        axes.plot(val_cost_list, color='m', label='Validation')
        axes.set_title('Cost Function Vs Learning Iteration')
        plt.xlabel('Learning Iteration')
        plt.ylabel('Cost function')
        plt.legend()
        axes.set_title('Cost Function Vs. Learning Iteration Number with ')
        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.savefig('Cost Vs. Iteration 1 feature', format='pdf')

    def get_classifier_tensors(data):
        nfeatures = 1
        nsamples = len(data)
        y_train = torch.zeros(nsamples, 2)
        x_train = torch.zeros(nsamples, nfeatures)
        for i in range(nsamples):
            material_st = data[i]['initial_structure']
            if data[i]['band_gap'] == 0.0:
                y_train[i][1] = 1
            else:
                y_train[i][0] = 1
            avg_cn = 0
            material_st_reduced = st.IStructure.get_primitive_structure(material_st)
            all_sites_neighbors = st.IStructure.get_all_neighbors(material_st_reduced, 3)
            nsites = len(material_st_reduced)
            if nsites == 10131:
                print(material_st_reduced)
            arr = [len(l) for l in all_sites_neighbors]
            avg_cn = np.mean(arr)
            x_train[i][0] = avg_cn
        return y_train, x_train

    # mp_data = m.query(criteria={"e_above_hull": 0}, properties=["band_gap", "initial_structure"])
    # mp_data = m.query(criteria={"$and": [{"nelements": 2}, {"e_above_hull": 0}]}, properties=["band_gap", "initial_structure"])
    # mp_data = m.query(criteria={'material_id': 'mp-1605'}, properties=["band_gap", "density", "initial_structure"])
    # pkl.dump(mp_data, open("mp_data.p", "wb"))
    mp_data_pkl = pkl.load(open("/Users/yair_levy_personal/PycharmProjects/pythonProject1/mp_data.p", "rb"))
    # my_material = mp_data_pkl[11]
    # pkl.dump(my_material, open("my_material.p", "wb"))
    # my_material_pkl = pkl.load(open("my_material.p", "rb"))
    # print(type(my_material_pkl['initial_structure'].types_of_specie[0]))
    # st.Structure.types_of_specie
    y, x = get_classifier_tensors(mp_data_pkl)
    x_train, x_small, y_train, y_small = train_test_split(x, y, random_state=20, test_size=0.6)
    x_val, x_test, y_val, y_test = train_test_split(x_small, y_small, random_state=20, test_size=0.5)
    # print('x_train: \n', x_train)
    # print('x_test: \n', x_test)
    # print('y_train: \n', y_train)
    # print('y_test: \n', y_test)
    # poscar = Poscar(materialStructure)
    # st.Structure.ntypesp , intresting property to get the number of types of elements in a molecule
    my_machine, train_cost_list, val_cost_list = bgc1.train(x_train, y_train, x_val, y_val)
    fig, axes = plt.subplots(1, 1)
    plot_cost(axes, fig, train_cost_list, val_cost_list)
    bgc1.test(my_machine, x_test, y_test)
