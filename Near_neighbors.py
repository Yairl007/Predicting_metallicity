import pickle as pkl
import pymatgen.analysis.local_env as env
import pymatgen.core.structure as stu
from pymatgen.ext.matproj import MPRester
with MPRester("sLhOEz6qQG8r4Tym") as m:

    mp_data = m.query(criteria={'material_id': 'mp-571576'}, properties=["initial_structure"])
    #mp_data_pkl = pkl.load(open("mp_data.p", "rb"))
    #my_material = mp_data_pkl[1010]
    st = mp_data[0]["initial_structure"]
    new_st = stu.IStructure.get_primitive_structure(st)
    all_neighbors = stu.IStructure.get_all_neighbors(new_st, 5)
    print(all_neighbors)
    print(type(all_neighbors))
    print(len(st))
    print(len(new_st))
    # mynn = env.CrystalNN()
    # nearest_neighbors = mynn.get_all_nn_info(st)
    # print(len(nearest_neighbors[0]))
    #cn_sum = 0
    # for i in range(len(st)):
    #    cn_sum += len(nearest_neighbors[i])
    # av_cn = cn_sum/len(nearest_neighbors)
    # print(len(nearest_neighbors))
