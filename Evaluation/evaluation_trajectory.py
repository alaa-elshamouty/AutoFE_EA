import os
import pickle

from utilities import get_opr_name

# read trajectories
duplicated_oprs_all = []
main_directory = '../results_backup/'

one_time_oprs = ['RandomTreesEmbedding',
                 'RBFSampler',
                 'QuantileTransformer']


def get_operators(trajectory, illegal_duplicate_oprs=[], seen_oprs=[]):
    if trajectory == (None, None, None, None, None):
        return seen_oprs, illegal_duplicate_oprs
    traj_current_member = trajectory[0]
    traj_first_member = trajectory[1]
    traj_second_member = trajectory[2]
    opr = get_opr_name(traj_current_member[0])
    if opr not in seen_oprs:
        seen_oprs.append(opr)
    else:
        if opr not in illegal_duplicate_oprs and opr in one_time_oprs:
            illegal_duplicate_oprs.append(opr)
    if traj_first_member == (None, None, None, None, None) and (
            traj_second_member == (None, None, None, None, None) or traj_second_member == None):
        return seen_oprs, illegal_duplicate_oprs
    # seen_oprs, duplicate_oprs = get_operators(traj_first_member, duplicate_oprs, seen_oprs)
    get_operators(traj_first_member, illegal_duplicate_oprs, seen_oprs)
    if traj_second_member:
        # seen_oprs, duplicate_oprs = get_operators(traj_second_member, duplicate_oprs, seen_oprs)
        get_operators(traj_second_member, illegal_duplicate_oprs, seen_oprs)
    return seen_oprs, illegal_duplicate_oprs


def get_duplicate_oprs(oprs):
    newlist = []  # empty list to hold unique elements from the list
    duplist = []  # empty list to hold the duplicate elements from the list
    for opr in oprs:
        if opr not in newlist:
            newlist.append(opr)
        else:
            duplist.append(opr)  # this method catches the first
    return duplist


for result_folder in os.listdir(main_directory):
    sub_directory = os.path.join(main_directory, result_folder, 'results')
    for dataset_folder in os.listdir(sub_directory):
        result_dict_path = os.path.join(sub_directory, dataset_folder)
        for filename in os.listdir(result_dict_path):
            if 'results' in filename:
                print(filename)
                path = os.path.join(result_dict_path, filename)
                with open(path, 'rb') as f:
                    result_dict = pickle.load(f)
                    trajectory = result_dict['trajectory_found']
                    seen_oprs, illegal_duplicated_oprs = get_operators(trajectory)
                    if len(illegal_duplicated_oprs) > 0:
                        duplicated_oprs_all.extend(illegal_duplicated_oprs)

import plotly.express as px
import pandas as pd
df = pd.DataFrame(data={
    'opr':duplicated_oprs_all
})
fig = px.histogram(df, x="opr")
fig.write_html('illegal_duplicate_oprs_count.html')
