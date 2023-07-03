import os
import pickle

import pandas as pd
import plotly.graph_objects as go
import time

from plotly.subplots import make_subplots

from Evaluation.plotly_setup import colors, font_dict, title_text
from data import global_datasets
from data.datasets_handling import get_dataset_name
from utilities import get_opr_name

# read trajectories
duplicated_oprs_all = []
main_directory = '../results_bo/'

one_time_oprs = ['RandomTreesEmbedding',
                 'RBFSampler',
                 'QuantileTransformer']


def plot_trajectory():
    directory = '../results_bo'
    fig = go.Figure()
    #datasets = [15,1510,18,469,37,54,23]
    datasets = [22]
    #datasets = [16,469, 1510]
    #ids = [2,0,0]
    #ids = [0]
    #datasets = global_datasets.datasets
    #ids = [0]*len(datasets)
    for j in range(len(datasets)):
        dataset_fn = datasets[j]
        dataset_name = get_dataset_name(dataset_fn)
        trajectory_file = os.path.join(directory, dataset_name, f'{dataset_name}_evaluation_traj_new.pkl')
        try:
            with open(trajectory_file, 'rb') as f:
                trajectory = pickle.load(f)
            #i = ids[j]
            for i in [1]:#range(len(trajectory)):
                dataset_name_plot = f'{dataset_name}'
                data = trajectory[i]
                opr_names = data['opr_names']
                train_scores = [score*100 for score in data['train_scores']]
                test_scores = [score*100 for score in data['test_scores']]

                fig.add_trace(go.Scatter(x=list(range(len(test_scores))), y=test_scores,
                                         mode='lines',
                                         line=dict(color=colors[j]),
                                         name=f'{dataset_name_plot}_test',
                                         hovertemplate=
                                         '<br>Step: %{x}<br>' +
                                         '<br>Test acc: %{y}<br>' +
                                         'Opr: %{text}',
                                         text=opr_names
                                         ))
                fig.add_trace(go.Scatter(x=list(range(len(train_scores))), y=train_scores,
                                         mode='lines',
                                         line=dict(shape='linear', dash='dot', color=colors[j]),
                                         name=f'{dataset_name_plot}_valid',
                                         hovertemplate=
                                         '<br>Step: %{x}<br>' +
                                         '<br>Train acc: %{y}<br>' +
                                         'Opr: %{text}',
                                         text=opr_names
                                         ))
        except Exception:
            continue
    fig.update_yaxes(
        title_text='Accuracy (%)',  # axis label
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
        title_text='Step',
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
    fig.update_layout(title=f'Trajectory of Dataset {dataset_name}',
                      template="simple_white",
                      legend=dict(
                          yanchor="top",
                          y=0.7,
                          xanchor="right",
                          x=0.9,
                          bordercolor="Black",
                          borderwidth=1,
                          bgcolor="rgba(255,255,255,0)",
                          font=font_dict
                      ),
                      font=font_dict
                      )
    #image_name='trajectory_overfitting_paper'
    image_name='trajectory_22_train_only_paper'
    fig.write_html(f'{image_name}.html')
    # fig.write_image(f'paper/{image_name}.pdf', width=2311, height=1202, scale=1)
    # time.sleep(2)
    # fig.write_image(f'paper/{image_name}.pdf', width=2311, height=1202, scale=1)
    print()


def analyze_opt_traj():
    dataset = '11'
    traj_file = os.path.join(main_directory, dataset, f'{dataset}_trajs.pkl')
    with open(traj_file, 'rb') as f:
        trajectory = pickle.load(f)


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


#
# for result_folder in os.listdir(main_directory):
#     sub_directory = os.path.join(main_directory, result_folder, 'results')
#     for dataset_folder in os.listdir(sub_directory):
#         result_dict_path = os.path.join(sub_directory, dataset_folder)
#         for filename in os.listdir(result_dict_path):
#             if 'results' in filename:
#                 print(filename)
#                 path = os.path.join(result_dict_path, filename)
#                 with open(path, 'rb') as f:
#                     result_dict = pickle.load(f)
#                     trajectory = result_dict['trajectory_found']
#                     seen_oprs, illegal_duplicated_oprs = get_operators(trajectory)
#                     if len(illegal_duplicated_oprs) > 0:
#                         duplicated_oprs_all.extend(illegal_duplicated_oprs)

# import plotly.express as px
# import pandas as pd
#
# df = pd.DataFrame(data={
#     'opr': duplicated_oprs_all
# })
# fig = px.histogram(df, x="opr")
# fig.write_html('illegal_duplicate_oprs_count.html')
def plot_ea_params():
    pop_size_df = pd.read_csv('wandb_data/pop_size.csv')
    dims_df = pd.read_csv('wandb_data/dims.csv')
    avg_fitness =pd.read_csv('wandb_data/avg_fitness.csv')
    best_member_fitness = pd.read_csv('wandb_data/valid_score.csv')
    fig = make_subplots(rows=2, cols=2)

    fig.add_trace(go.Scatter(x=list(range(len(pop_size_df))), y=pop_size_df['23 - pop_size'],
                             mode='lines',
                             line=dict(color=colors[0]),
                             name=f'Population Size',
                             ))

    fig.add_trace(
        go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
        row=1, col=2
    )

    fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    fig.show()


def get_opr_count():
    directory = '../results_bo'
    datasets = global_datasets.datasets
    oprs_total_count={}
    oprs_dec_count = {}
    oprs_inc_count={}
    for j in range(len(datasets)):
        dataset_fn = datasets[j]
        dataset_name = get_dataset_name(dataset_fn)
        trajectory_file = os.path.join(directory, dataset_name, f'{dataset_name}_evaluation_traj_new.pkl')
        try:
            with open(trajectory_file, 'rb') as f:
                trajectory = pickle.load(f)
            for i in range(len(trajectory)):
                dataset_name_plot = f'{dataset_name}_{i}'
                data = trajectory[i]
                opr_names = [name.split('_')[0].split('(')[0] for name in data['opr_names']]
                test_scores = [score * 100 for score in data['test_scores']]
                for z in range(len(test_scores)-1):
                    opr = opr_names[z+1]
                    update_dict_count(opr, oprs_total_count)
                    if test_scores[z+1] - test_scores[z] > 0:
                        update_dict_count(opr, oprs_inc_count)
                    elif test_scores[z+1] - test_scores[z] < 0:
                        update_dict_count(opr, oprs_dec_count)
        except Exception:
            continue
    final_dict = {
        'total': oprs_total_count,
        'inc':oprs_inc_count,
        'dec':oprs_dec_count,
    }
    with open('opr_counts.pickle', 'wb') as handle:
        pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def update_dict_count(opr, oprs_count):
    if opr in oprs_count:
        oprs_count[opr] = oprs_count[opr] + 1
    else:
        oprs_count[opr] = 1


def plot_operations_used():
    with open('opr_counts.pickle', 'rb') as handle:
         oprs_count= pickle.load(handle)
    fig = go.Figure()
    total = oprs_count['total']
    total.pop('check')
    total.pop('None')
    x = list(total.keys())
    inc = oprs_count['inc']
    dec = oprs_count['dec']
    same = {name:(total[name] - ((inc[name] if name in inc else 0)  + (dec[name] if name in dec else 0))) for name in x}
    fig.add_trace(go.Bar(
        x=x,
        y=[inc[name] if name in inc else 0 for name in x],
        name='Increase Count',
        marker_color=colors[2]
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=[dec[name] if name in dec else 0 for name in x ],
        name='Decrease Count',
        marker_color=colors[1]
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=[same[name] for name in x],
        name='No effect Count',
        marker_color=colors[4]
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=[total[name] for name in x],
        name='Total Count',
        marker_color=colors[0]
    ))

    fig.update_yaxes(
        title_text='Count',  # axis label
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
        title_text='Operation Name',
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
    fig.update_layout(title=f'Operation Count',
                      template="simple_white",
                      legend=dict(
                          yanchor="top",
                          y=0.8,
                          xanchor="right",
                          x=0.7,
                          bordercolor="Black",
                          borderwidth=1,
                          bgcolor="rgba(255,255,255,0)",
                          font=font_dict
                      ),
                      font=font_dict
                      )
    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=-45)
    image_name = 'opr_count_paper_new'
    fig.write_html(f'{image_name}.html')


if __name__ == '__main__':
    plot_trajectory()
    #plot_ea_params()
    #get_opr_count()
    #plot_operations_used()