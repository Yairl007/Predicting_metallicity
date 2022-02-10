import matplotlib.pyplot as plt
import csv
import math
import numpy as np
import pandas as pd
import pickle as pkl
import band_gap_pymatgen as bgp
from pymatgen.ext.matproj import MPRester
with MPRester("sLhOEz6qQG8r4Tym") as m:
    def dict_to_csv(mp_data,file_name):
        keys = mp_data[0].keys()
        with open(str(file_name)+'.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(mp_data)
        return output_file.name

    def plot_csv_hist(csv_fileName : str, nbins = 50):
        # property.value_counts().sort_index().to_csv('band_gaps_count.csv')
        all_csv = pd.read_csv(csv_fileName, quoting=2)
        nkeys = all_csv.keys().size
        fig, ax = plt.subplots(1,nkeys)
        i = 0
        for key in all_csv.keys():
            print(all_csv[key].value_counts().sort_index())
            plt.hist(all_csv[all_csv['band_gap'] == 0.0], all_csv[all_csv['band_gap'] > 0.0], bins=nbins)
            # all_csv[key].hist(bins=nbins)
            plt.xlabel(str(key))
            plt.ylabel('counts')
            plt.title('Number of Elements Histogram')
            i += 1
        plt.show()

    def obtain_bandgap_data():
        mp_data = m.query(criteria={"e_above_hull": 0}, properties=["band_gap"])
        #mp_data = m.query(criteria={'material_id': 'mp-1605'}, properties=['pretty_formula',"initial_structure"])
        #print(mp_data,'\n', type(mp_data[0]['initial_structure']))
        keys = mp_data[0].keys()
        with open('band_gaps.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(mp_data)

    def sigmoid(x):
        a = []
        for item in x:
            a.append(1 / (1 + math.exp(-item)))
        return a


    x = np.arange(-10., 10., 0.2)
    sig = sigmoid(x)
    plt.plot(x, sig)
    plt.title("The Sigmoid Function", size=18)
    plt.xlabel("Z", size=14)
    plt.ylabel("g(z)", size=14)
    plt.show()
#with open("Target tensors 5Feb.p", 'rb') as y, open("Features tensors 5Feb.p", 'rb') as x:

    #x_array = np.array(x_tensors_pkl)
    #x_df = pd.DataFrame(x_array)
    #x_df.to_csv('features_CSV.csv')
    #mp_data = m.query(criteria={"e_above_hull": 0}, properties=["initial_structure"])
    #file_name = dict_to_csv(mp_data,"initial_structures")
    #plot_csv_hist('nelements.csv')
    mp_data = m.query(criteria={'material_id': 'mp-556442'}, properties=["initial_structure","band_gap"])
