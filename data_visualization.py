import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
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
            all_csv[key].hist(bins=nbins)
            plt.xlabel(str(key))
            plt.ylabel('counts')
            plt.title('Number of Elements Histogram')
            i += 1
        plt.show()

    def obtain_data():
        mp_data = m.query(criteria={"e_above_hull": 0}, properties=["band_gap"])
        #mp_data = m.query(criteria={'material_id': 'mp-1605'}, properties=['pretty_formula',"initial_structure"])
        #print(mp_data,'\n', type(mp_data[0]['initial_structure']))
        keys = mp_data[0].keys()
        with open('band_gaps.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(mp_data)


    #mp_data = m.query(criteria={"e_above_hull": 0}, properties=["nelements"])
    #file_name = dict_to_csv(mp_data,"nelements")
    plot_csv_hist('nelements.csv')
