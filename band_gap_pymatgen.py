# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.
import torch
import numpy as np
import pickle as pkl
from pymatgen.ext.matproj import MPRester
import band_gap_classifier as bgc
from sklearn.model_selection import train_test_split
import csv
import pymatgen.analysis.local_env as env
import matplotlib.pyplot as plt
from pymatgen.analysis.local_env import NearNeighbors as nn
import numpy.ma as ma
import pandas as pd
import pymatgen.core.periodic_table as periodic
import pymatgen.core.structure as st
import pymatgen.core.lattice as lat
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
        nfeatures = 3
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
            # feature 2: minimal bond length
            if material_st.num_sites == 1:
                material_st.make_supercell(3)
                all_bonds = material_st.distance_matrix
                min_bond = np.min(material_st.distance_matrix[np.nonzero(all_bonds)])
            else:
                all_bonds = material_st.distance_matrix
                min_bond = np.min(material_st.distance_matrix[np.nonzero(all_bonds)])
                location = np.where(all_bonds == min_bond)
                atom1radius = material_st.sites[location[0][0]].specie.atomic_radius
                atom2radius = material_st.sites[location[0][1]].specie.atomic_radius
                min_bond_norm = min_bond/(atom1radius+atom2radius)
                st.IStructure.sites
            x_train[i][1] = min_bond_norm
            # feature 3: metal to non metal ratio
            x_train[i][2] = 1
            non_metals = set(material_st.atomic_numbers).intersection(
                {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 34, 35, 36, 53, 54, 86})
            if len(non_metals) != 0:
                for element_comp in list(material_st.composition.fractional_composition.items()):
                    if element_comp[0].number in non_metals:
                        x_train[i][2] -= element_comp[1]
            lat._compute_cube_index()
            # feature 4: average coordination number - avg_cn
            avg_cn = 0
            # material_st_reduced = st.IStructure.get_primitive_structure(material_st)
            # all_sites_neighbors = st.IStructure.get_all_neighbors(material_st_reduced, 5)
            #nsites = len(material_st_reduced)
            # arr = [len(l) for l in all_sites_neighbors]
            # avg_cn = np.mean(arr)
            # x_train[i][3] = avg_cn
        return y_train, x_train


    def plot_3D_features(x,y):
        """
        plotting function to help visualize the features and find trends in them.
        x - tensor of features with shape - n*m, m the number of features n the number of samples.
        y - target values vector with shape n*2 where y[i,1] is the probability the i^th sample is conductive- band gap zero
        outputs a figure with conducting samples in orange and insulating samples in blue
        """
        mask_y = y[:, 1] == 1
        x_conductor = x[mask_y, :]
        x_insulator = x[~mask_y, :]
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(x_conductor[:, 0], x_conductor[:, 1], x_conductor[:, 2], color='orange')
        ax.scatter(x_insulator[:, 0], x_insulator[:, 1], x_insulator[:, 2], color='blue')
        ax.set_xlabel('density [gr/cm^3]')
        ax.set_ylabel('minimal bond length')
        ax.set_zlabel('metal to non metal ratio')
        plt.show()
    # pkl.dump(mp_data, open("mp_data.p", "wb"))
    # mp_data_pkl = pkl.load(open("mp_data.p", "rb"))
    # my_material = mp_data_pkl[11]
    # pkl.dump(my_material, open("my_material.p", "wb"))
    # my_material_pkl = pkl.load(open("my_material.p", "rb"))
    # print(type(my_material_pkl['initial_structure'].types_of_specie[0]))
    # st.Structure.types_of_specie
    # y, x = get_classifier_tensors(mp_data_pkl)
    # pkl.dump(x, open("features tensors.p", "wb"))
    # pkl.dump(y, open("Target tensors.p", "wb"))
    x_tensors_pkl = pkl.load(open("/Users/yair_levy_personal/PycharmProjects/Predicting_metallicity1/features tensors.p", "rb"))
    y_tensors_pkl = pkl.load(open("/Users/yair_levy_personal/PycharmProjects/Predicting_metallicity1/Target tensors.p", "rb"))
    x_train, x_small, y_train, y_small = train_test_split(x_tensors_pkl, y_tensors_pkl, random_state=20, test_size=0.6)
    x_val, x_test, y_val, y_test = train_test_split(x_small, y_small, random_state=20, test_size=0.5)
    plot_3D_features(x_tensors_pkl, y_tensors_pkl)
    # print('x_train: \n', x_train)
    # print('x_test: \n', x_test)
    # print('y_train: \n', y_train)
    # print('y_test: \n', y_test)
    # poscar = Poscar(materialStructure)
    # st.Structure.ntypesp , intresting property to get the number of types of elements in a molecule
    my_machine, train_cost_list, val_cost_list = bgc.train(x_train, y_train, x_val, y_val)
    fig, axes = plt.subplots(1, 1)
    plot_cost(axes, fig, train_cost_list, val_cost_list)
    bgc.test(my_machine, x_test, y_test)
