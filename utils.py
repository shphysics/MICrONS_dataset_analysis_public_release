from caveclient import CAVEclient
import pandas as pd
import numpy as np
import scipy.spatial as sci_spatial
from scipy.spatial import distance_matrix
from tqdm import tqdm
import csv
import pickle
from collections import defaultdict

def testing_func():
    """
    for testing purpose
    """
    print("utils is properly imported")

def creating_syp_information(valid_ids, verified_ids_len, client):
    """
    creating 1. syp_dict 2. syp_positions (n by 3 matrix) 3. syp_pos_tracking that has the syp id
    """
    pt_root_ids = valid_ids
    # syp_dict = {}
    syp_dict = defaultdict(list)
    syp_positions = np.zeros(3) ##this will be the voxel positions
    syp_pos_tracking = []
    syp_unique_id = set()
    with tqdm(total= verified_ids_len) as pbar:
        for i in range(verified_ids_len):
            pbar.update(1)
            # the seg_id is the id of the verified pre-syp neurons
            seg_id = pt_root_ids[i]
            seg_id_int = seg_id.item()

            # outputs of the selected neuron (the selected neurons will be the pre-syp neurons)
            output_df = client.materialize.synapse_query(pre_ids=seg_id)
            post__pt_root_ids = output_df.post_pt_root_id
            ser_post__pt_root_ids = pd.Series(post__pt_root_ids) #post neurons
            output_syp_ids = output_df.id
            ser_output_syp_ids = pd.Series(output_syp_ids) #syp to connect curr seg with the post neurons
            output_syps_pos = output_df.ctr_pt_position
            ser_output_syps_pos = pd.Series(output_syps_pos) #syp ctr pos in voxel

            output_post_zipping_list = zip(ser_post__pt_root_ids, ser_output_syp_ids, ser_output_syps_pos)
            for (post_pt_root_id, output_syp_id, output_syp_pos) in output_post_zipping_list:
    #             if not output_syp_id in syp_pos_tracking: #this is not necessary (but just do a sanity check first)
                syp_pos_tracking.append(output_syp_id)
                syp_positions = np.vstack((syp_positions, output_syp_pos))
                syp_unique_id.add(output_syp_id)
                syp_dict[output_syp_id].append((seg_id, post_pt_root_id))
    
    syp_dict = dict(syp_dict)
    return syp_dict, syp_positions, syp_pos_tracking

def save_syp_information(syp_dict, syp_voxel_pos, syp_pos_tracking):
    """
    save syp_dict, syp_voxel_pos, syp_xyz_pos (calculated based on voxel pos), and syp_pos_tracking
    """
    # save the syp_dict
    filename = 'syp_dict_unproof'
    outfile = open(filename, 'wb')
    pickle.dump(syp_dict, outfile)
    outfile.close()

    # verificiation of removal of place holder row and converting voxel position to xyz position
    print("verifying removal of place holder")
    print(syp_voxel_pos[0])
    print(syp_voxel_pos[1])
    syp_positions = np.delete(syp_voxel_pos, 0, 0)
    print(syp_positions[0])
    print(np.shape(syp_positions))
    print("verification completed")
    
    # save syp_voxel_pos
    filename2 = 'syp_voxel_pos_unproof'
    outfile2 = open(filename2, 'wb')
    pickle.dump(syp_positions, outfile2)
    outfile2.close()

    # calculate and save syp_xyz_pos
    conversion_array = np.array([4, 4, 40])
    syp_xyz_pos = np.empty_like(syp_positions)
    for i in range(np.shape(syp_positions)[0]):
        syp_xyz_pos[i] = np.multiply(syp_positions[i], conversion_array)

    filename3 = 'syp_xyz_pos_unproof'
    outfile3 = open(filename3, 'wb')
    pickle.dump(syp_xyz_pos, outfile3)
    outfile3.close()

    # save syp_pos_tracking
    filename4 = 'syp_pos_tracking_unproof'
    outfile4 = open(filename4, 'wb')
    pickle.dump(syp_pos_tracking, outfile4)
    outfile4.close()

def read_in_syp_information(syp_dict, syp_voxel_pos, syp_xyz_pos, syp_pos_tracking):
    """
    reading from files named 'syp_dict', 'syp_voxel_pos', 'syp_xyz_pos', and 'syp_pos_tracking'
    the arguments of this function are set to equal to the file names as specified in the previous sentence
    """
    infile_syp_dict = open(syp_dict, 'rb')
    syp_dict = pickle.load(infile_syp_dict)
    infile_syp_dict.close()

    infile_syp_voxel_pos = open(syp_voxel_pos, 'rb')
    syp_voxel_pos = pickle.load(infile_syp_voxel_pos)
    infile_syp_voxel_pos.close()

    infile_syp_xyz_pos = open(syp_xyz_pos, 'rb')
    syp_xyz_pos = pickle.load(infile_syp_xyz_pos)
    infile_syp_xyz_pos.close()

    infile_syp_pos_tracking = open(syp_pos_tracking, 'rb')
    syp_pos_tracking = pickle.load(infile_syp_pos_tracking)
    infile_syp_pos_tracking.close()

    return syp_dict, syp_voxel_pos, syp_xyz_pos, syp_pos_tracking

def save_obj_with_name(obj, name):
    """
    save the python obj in a pickle file format with a specified file name; 'name' should be a string
    """
    outfile = open(name,'wb')
    pickle.dump(obj, outfile)
    outfile.close()

def load_obj_from_filename(name):
    """
    load the python pickle obj into the program; 'name' in the argument is a string type of filename
    """
    infile = open(name, 'rb')
    obj = pickle.load(infile)
    infile.close()
    return obj


def get_seq_with_length_n_of_unique_keys_for_occur_more_than_m_times_w_type_lst(n, m, dict_with_certain_type):
    seq_keys_for_more_than_n_occur_lst = []
    seq_keys_for_more_than_n_occur = filter(lambda key: len(key) == n and len(dict_with_certain_type[key]) > m,
                                            dict_with_certain_type)
    for key in seq_keys_for_more_than_n_occur:
        seq_keys_for_more_than_n_occur_lst.append(key)

    unique_neurons_with_more_n_occur_w_type_lst = {}
    for key in seq_keys_for_more_than_n_occur_lst:
        has_occurred = set()
        instance_tuple = key
        instance_list = list(instance_tuple)
        has_occurred.add(instance_list[0])
        for i in range(1, len(instance_list)):
            neuron_i = instance_list[i]
            if neuron_i in has_occurred:
                continue
            else:
                has_occurred.add(neuron_i)
        if len(has_occurred) == len(instance_tuple):
            unique_neurons_with_more_n_occur_w_type_lst[instance_tuple] = dict_with_certain_type[instance_tuple]
    return unique_neurons_with_more_n_occur_w_type_lst
