import torch
import numpy as np
import pickle as pkl
from pymatgen.ext.matproj import MPRester
import bgc_1_layer as bgcc
from sklearn.model_selection import train_test_split
import pymatgen.analysis.local_env as env
import matplotlib.pyplot as plt
from pymatgen.analysis.local_env import NearNeighbors as nn
import numpy.ma as ma
import pandas as pd
import pymatgen.core.periodic_table as periodic
import pymatgen.core.structure as st
with MPRester("sLhOEz6qQG8r4Tym") as m:
    # mp_data = m.query(criteria={"e_above_hull": 0}, properties=["band_gap", "initial_structure"])
    # mp_data = m.query(criteria={"$and": [{"nelements": 2}, {"e_above_hull": 0}]}, properties=["band_gap", "initial_structure"])
    # mp_data = m.query(criteria={'material_id': 'mp-1605'}, properties=["band_gap", "density", "initial_structure"])

    def plot_cost(axes, fig, train_cost_list, val_cost_list):
        axes.cla()
        axes.plot(train_cost_list, color='darkslategrey', label='Training')
        axes.plot(val_cost_list, color='m', label='Validation')
        axes.set_title('Cost Function Vs Learning Iteration')
        plt.xlabel('Learning Iteration')
        plt.ylabel('Cost function')
        plt.legend()
        axes.set_title('Cost Function Vs. Learning Iteration Number')
        fig.show()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def get_classifier_tensors(data):
        nfeatures = 2
        nsamples = len(data)
        y_train = torch.zeros(nsamples, 2)
        x_train = torch.zeros(nsamples, nfeatures)
        for i in range(nsamples):
            material_st = data[i]['initial_structure']
            if data[i]['band_gap'] == 0.0:
                y_train[i][1] = 1
            else:
                y_train[i][0] = 1
            # feature 1: density
            x_train[i][0] = material_st.density
            # feature 3: metal to non metal ratio
            x_train[i][1] = 1
            non_metals = set(material_st.atomic_numbers).intersection(
                {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 34, 35, 36, 53, 54, 86})
            if len(non_metals) != 0:
                for element_comp in list(material_st.composition.fractional_composition.items()):
                    if element_comp[0].number in non_metals:
                        x_train[i][1] -= element_comp[1]
        return y_train, x_train


    # pkl.dump(mp_data, open("mp_data.p", "wb"))
    mp_data_pkl = pkl.load(open("mp_data.p", "rb"))
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
    my_machine, train_cost_list, val_cost_list = bgcc.train(x_train, y_train, x_val, y_val)
    fig, axes = plt.subplots(1, 1)
    plot_cost(axes, fig, train_cost_list, val_cost_list)
    bgcc.test(my_machine, x_test, y_test)
