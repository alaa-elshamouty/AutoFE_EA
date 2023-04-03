import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import plotly.figure_factory as ff

from plotly.subplots import make_subplots

directory = '../results/'
#directory = '../results_backup/results_backup_24_03_23_14_00/results/'


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
        'best_member':best_member,
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
        df = pd.read_csv('results_before_after_df.csv')
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
    df = pd.read_csv('results_before_after_df.csv')
    names = df['name'].values

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


def plot_distribution_per_dataset(dataset_name, X_train_before, X_train_after, X_test_before, X_test_after):
    dir = dataset_name
    print(min(X_train_before.shape[-1], X_train_after.shape[-1]))
    for feature_id in range(min(X_train_before.shape[-1], X_train_after.shape[-1])):
        # Add histogram data
        x1 = X_train_before[:, feature_id]
        x2 = X_test_before[:, feature_id]
        x3 = X_train_after[:, feature_id]
        x4 = X_test_after[:, feature_id]

        # Group data together
        hist_data = [x1, x2, x3, x4]

        group_labels = ['Train Before: {}'.format(result_dict['train_acc_before']),
                        'Test Before: {}'.format(result_dict['test_acc_before']),
                        'Train After: {}'.format(result_dict['best_member_fitness']),
                        'Test After: {}'.format(result_dict['test_acc_after'])]

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
        fig.update_layout(title_text=str(result_dict['trajectory_found']), height=900,
                          font=dict(
                              family="Courier New, monospace",
                              size=18,
                              color="RebeccaPurple"
                          )
                          )
        fig.write_html("{}/distribution_f{}.html".format(dir, feature_id))


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


#plot_fitness_traj()
#plot_bar_diff()
#opt_config_parallel_coords()
print("dummy for debug")
plot_before_after_acc()
plot_bars_before_after()
exit()
folder = '31'
X_train_before, X_train_after, X_test_before, X_test_after, result_dict = get_dataset_train_test(
    directory + folder)
traj = result_dict['trajectory_found']
if not os.path.exists(folder):
    os.makedirs(folder)
file = folder + '/traj.txt'
f = open(file, 'w')
for t in traj:
    if t is None:
        t = 'None'
    f.write(' '.join(str(s) for s in t) + '\n')
plot_distribution_per_dataset(folder, X_train_before, X_train_after, X_test_before, X_test_after)
