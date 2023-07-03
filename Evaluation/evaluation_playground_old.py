import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import plotly.figure_factory as ff
import time
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

from EA.strategies import Recombination, Mutation
from Evaluation.plotly_setup import colors, font_dict
from data import global_datasets
from data.datasets_handling import load_dataset
from utilities import get_opr_name

directory = '../results_bo/results/'


# directory = '../results_backup/results_backup_24_03_23_14_00/results/'


def get_before_after_acc():
    before_train = []
    best_member = []
    after_train = []
    before_test = []
    after_test = []
    trajs = []
    fitness_trajs = []
    names = []
    for dataset_folder in os.listdir(directory):
        for filename in os.listdir(directory + dataset_folder):
            if 'results' in filename:
                path = os.path.join(directory, dataset_folder, filename)
                with open(path, 'rb') as f:
                    loaded_dict = pickle.load(f)
                    before_train.append((loaded_dict['train_acc_before']))
                    best_member.append(loaded_dict['best_member_fitness'])
                    after_train.append(loaded_dict['train_acc_after'])
                    before_test.append(loaded_dict['test_acc_before'])
                    after_test.append((loaded_dict['test_acc_after']))
                    trajs.append(loaded_dict['trajectory_found'])
                    fitness_trajs.append((loaded_dict['fitness_trajectory']))
                    names.append(filename)
                    break
    df = pd.DataFrame(data={
        'before_train': before_train,
        'best_member': best_member,
        'after_train': after_train,
        'before_test': before_test,
        'after_test': after_test,
        'trajs': trajs,
        'fitness_traj': fitness_trajs,
        'name': names
    })
    df.to_csv('results_before_after_df.csv')

    return before_train, best_member, after_train, before_test, after_test, trajs, fitness_trajs


def plot_before_after_acc(load_file=False):
    if load_file:
        df = pd.read_csv('old/results_before_after_df.csv')
    else:
        before_train, best_member, after_train, before_test, after_test, trajs, fitness_trajs = get_before_after_acc()
        df = pd.DataFrame(data={
            'before_train': before_train,
            'best_member': best_member,
            'after_train': after_train,
            'before_test': before_test,
            'after_test': after_test,
            'trajs': trajs
        })

    fig_test_acc = px.scatter(df, x='before_test', y='after_test')
    fig_train_acc = px.scatter(df, x='before_train', y='after_train',
                               color_discrete_sequence=px.colors.qualitative.Antique)
    fig_line = px.line([0, 1], [0, 1])
    fig_line.update_traces(patch={"line": {"color": "gray", "width": 4, "dash": 'dot'}})
    fig = go.Figure(data=fig_test_acc.data + fig_line.data + fig_train_acc.data)
    fig['data'][0]['name'] = 'Test Accuracy'
    fig['data'][2]['name'] = 'Train Accuracy'
    fig['data'][0]['showlegend'] = True
    fig['data'][2]['showlegend'] = True
    fig.update_layout(
        xaxis_title="Accuracy before Feature Engineering", yaxis_title="Accuracy after Feature Engineering")

    fig.write_html("before_after_plot.html")


def plot_bars_before_after():
    df = pd.read_csv('old/results_before_after_df.csv')
    names = [str(i) + name for i, name in enumerate(df['name'].values)]

    fig = go.Figure(data=[
        go.Bar(name='Train before EA', x=names, y=df['before_train']),
        go.Bar(name='Train After EA', x=names, y=df['after_train']),
        go.Bar(name='Test Before EA', x=names, y=df['before_test']),
        go.Bar(name='Test After EA', x=names, y=df['after_test']),
        go.Bar(name='Member Best Fitness', x=names, y=df['best_member'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.write_html("before_after_barplot.html")


def get_dataset_train_test(dataset_folder):
    info = {}
    results_name = [name for name in os.listdir(dataset_folder) if 'results' in name]
    path = os.path.join(dataset_folder, results_name[0])
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    if 'normalized_X_train.npy' in os.listdir(dataset_folder):
        info['normalized'] = True
        X_train_before = np.load(os.path.join(dataset_folder, 'normalized_X_train.npy'))
        X_test_before = np.load(os.path.join(dataset_folder, 'normalized_X_test.npy'))
    else:
        info['normalized'] = False
        X_train_before = np.load(os.path.join(dataset_folder, 'X_train.npy'))
        X_test_before = np.load(os.path.join(dataset_folder, 'X_test.npy'))

    X_train_after = np.load(os.path.join(dataset_folder, 'X_train_after_trajectory.npy'))
    X_test_after = np.load(os.path.join(dataset_folder, 'X_test_after_trajectory.npy'))
    return X_train_before, X_train_after, X_test_before, X_test_after, loaded_dict


def plot_distributions():
    fig = make_subplots(rows=len(os.listdir(directory)))
    for i, dataset_folder in enumerate(os.listdir(directory)):
        if dataset_folder + '_results.pkl' in os.listdir(directory + dataset_folder):
            X_train_before, best_member, X_train_after, X_test_before, X_test_after = get_dataset_train_test(
                directory + dataset_folder)
            plot_distribution_per_dataset(X_train_before, X_train_after, X_test_before, X_test_after)
    fig.write_html("distributions.html")


def plot_distribution_per_dataset(dataset, X_train_before, X_train_after, X_test_before, X_test_after):
    # print(min(X_train_before.shape[-1], X_train_after.shape[-1]))
    color = colors[:4]
    for feature_id in range(min(X_train_before.shape[-1], X_train_after.shape[-1])):
        # Add histogram data
        x1 = X_train_before[:, feature_id]
        x2 = X_test_before[:, feature_id]
        x3 = X_train_after[:, feature_id]
        x4 = X_test_after[:, feature_id]

        # Group data together
        hist_data_before = [x1, x2]
        hist_data_after = [x3, x4]

        group_labels_before = ['Train Before',
                               'Test Before']
        group_labels_after = ['Train After',
                              'Test After']

        fig_before = ff.create_distplot(hist_data_before, group_labels_before, bin_size=.2, colors=color)
        fig_after = ff.create_distplot(hist_data_after, group_labels_after, bin_size=.2, colors=color)
        for fig in [fig_before, fig_after]:
            fig.update_yaxes(
                title_text='',  # axis label
                showline=True,  # add line at x=0
                linecolor='black',  # line color
                linewidth=2.4,  # line size
                ticks='outside',  # ticks outside axis
                tickfont=font_dict,  # tick label font
                titlefont=font_dict,  # tick label font
                mirror='allticks',  # add ticks to top/right axes
                tickwidth=2.4,  # tick width
                tickcolor='black',  # tick color
                showgrid=True,
                automargin=True,
            )
            fig.update_yaxes(title_standoff=0)
            fig.update_yaxes(title_standoff=25)

            fig.update_xaxes(
                title_text=f'Feature ID {feature_id}',
                showline=True,
                # showticklabels=True,
                linecolor='black',
                linewidth=2.4,
                ticks='outside',
                tickfont=font_dict,
                mirror='allticks',
                tickwidth=2.4,
                tickcolor='black',
                showgrid=True,
                automargin=True

            )
            fig.update_layout(title=f'Distribution Shift After EA for Dataset ID {dataset}',
                              template="simple_white",
                              legend=dict(
                                  yanchor="top",
                                  y=0.8,
                                  xanchor="right",
                                  x=0.5,
                                  bordercolor="Black",
                                  borderwidth=1,
                                  bgcolor="rgba(255,255,255,0)",
                                  font=dict(family='Arial',
                                            size=20,
                                            color='black'
                                            )
                              ),
                              font=font_dict
                              )
        for image_name_info, fig in zip(['before', 'after'], [fig_before, fig_after]):
            image_name = f'dist/1510/distribution_d_{dataset}f_{feature_id}_{image_name_info}'
            fig.write_html(f'{image_name}.html')
            # fig.write_image(f'paper/{image_name}.pdf', width=2311, height=1202, scale=1)
            # time.sleep(2)
            # fig.write_image(f'paper/{image_name}.pdf', width=2311, height=1202, scale=1)


def plot_bar_diff():
    df = pd.read_csv("../results_backup/results_backup_27_03_23_14_45/evaluation/results_before_after_df.csv")
    df["train_diff"] = df['after_train'] - df["before_train"]
    df["test_diff"] = df['after_test'] - df["before_test"]
    train_diff_avg = df["train_diff"].average()

    names = df['name'].values
    fig = go.Figure(data=[
        go.Bar(name='Train Difference', x=names, y=df['train_diff']),
        go.Bar(name='Test Difference', x=names, y=df['test_diff'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.write_html("diff_barplot.html")


def opt_config_parallel_coords():
    df = pd.read_csv("opt_configs_combined.csv")
    fig = px.parallel_coordinates(df,
                                  dimensions=list(df.keys()),
                                  color_continuous_scale=px.colors.diverging.Tealrose,
                                  color_continuous_midpoint=2)
    fig.write_html("opt_configs.html")


def plot_fitness_traj():
    before_train, best_member, after_train, before_test, after_test, trajs, fitness_trajs = get_before_after_acc()
    fitness_df = pd.DataFrame(data={
        'steps': range(0, len(fitness_trajs[0])),
        'fitness': [score for opr, score in fitness_trajs[0]]
    })
    fig = px.line(fitness_df, x="steps", y="fitness", title='Fitness Trajectory')
    fig.write_html("fitness_traj.html")


def apply_operator(traj, data, partner_data, rec=False):
    new_x = data.copy()
    opr, _, col_id, _, partner_col_id = traj
    if rec:
        _, new_x = Recombination.apply_recombination(opr, new_x, col_id, partner_data, partner_col_id,
                                                     applying_traj=True)
    else:
        _, new_x = Mutation.apply_mutation(opr, new_x, col_id=col_id, applying_traj=True)
    return get_opr_name(opr), new_x


def apply_trajectory(dataset, trajectory):
    if trajectory == (None, None, None, None, None):
        return dataset
    traj_current_member = trajectory[0]
    traj_first_member = trajectory[1]
    traj_second_member = trajectory[2]
    if traj_first_member == (None, None, None, None, None) and traj_second_member == (None, None, None, None, None):
        opr, new_x = apply_operator(traj_current_member, dataset, dataset, rec=True)
        return new_x
    elif traj_first_member == (None, None, None, None, None) and not traj_second_member:
        opr, new_x = apply_operator(traj_current_member, dataset, None, rec=False)
        return new_x

    first_member_data = apply_trajectory(dataset, traj_first_member)
    if traj_second_member:
        second_member_data = apply_trajectory(dataset, traj_second_member)
        opr, new_x = apply_operator(traj_current_member, first_member_data, second_member_data, rec=True)
    else:
        opr, new_x = apply_operator(traj_current_member, first_member_data, None, rec=False)

    return new_x


# plot_fitness_traj()
# plot_bar_diff()
# opt_config_parallel_coords()
# print("dummy for debug")
# plot_before_after_acc()
# plot_bars_before_after()
# exit()
# X_train = np.load('results/1510/X_train_before_sin.npy')
# new_x_train = np.load('results/1510/X_train_after_sin.npy')
# X_test = np.load('results/1510/X_test_before_sin.npy')
# new_x_test = np.load('results/1510/X_test_after_sin.npy')
datasets_ = [469]#global_datasets.datasets
# plot_distribution_per_dataset(dataset,X_train, new_x_train, X_test, new_x_test)
# exit()
directory = '../results_bo'
new_features = []
for dataset in datasets_:
    X, y = load_dataset(id=dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    trajectory_file = os.path.join(directory, str(dataset), f'{dataset}_trajs.pkl')
    try:
        with open(trajectory_file, 'rb') as f:
            trajectory = pickle.load(f)
    except:
        new_features.append(np.NAN)
        continue

    for i in range(len(trajectory)):
        traj = trajectory[i]
        if traj:
        # new_x_train = apply_trajectory(X_train, traj)
            new_x_test = apply_trajectory(X_test, traj)
            new_features.append(new_x_test.shape[-1])
            break
    # plot_distribution_per_dataset(dataset, X_train, new_x_train, X_test, new_x_test)
    print('dummy')
df = pd.read_csv('../data/datasets.csv')
df['after_nr_features'] = new_features
df.to_csv('../data/datasets.csv')