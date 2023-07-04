# %%
"""
Compute statistics saved in RobotControlMetric dataholder
1. access to all monitor.pb files in exp_dir
2. for each 10 episodes: calculate the statistics
3. calculate mean and variance between the different chunks
4. Visualize the metrics
"""
import os
import time
import traceback
from os.path import join, dirname

import numpy as np
import tqdm

os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import pandas as pd
from plotly.graph_objs import Box, Figure, Scatter
from plotly.subplots import make_subplots
import plotly.express as px

pd.options.plotting.backend = "plotly"

from cara.workplaces.obstacles_workplace import ObstacleWorkplaceEnv

scenarios = [ObstacleWorkplaceEnv.SCENARIO_BINS,
             ObstacleWorkplaceEnv.SCENARIO_REGAL,
             ObstacleWorkplaceEnv.SCENARIO_PRIMITIVE_OBJECTS,
             ObstacleWorkplaceEnv.SCENARIO_REAL_OBJECTS,
             ]
scenario_names = {ObstacleWorkplaceEnv.SCENARIO_BINS: "Bins",
                  ObstacleWorkplaceEnv.SCENARIO_REGAL: "Shelves",
                  ObstacleWorkplaceEnv.SCENARIO_PRIMITIVE_OBJECTS: "Boxes Clutter",
                  ObstacleWorkplaceEnv.SCENARIO_REAL_OBJECTS: "Real Objects Clutter",
                  }

# =============
METRIC_SPEED = "speed / (m/s)"
METRIC_BOOL = "Boolean"
METRIC_DISTANCE = "(m)"
METRIC_TIME = "time / (s)"
METRIC_SMOOTHNESS = "(rad)"
y_axis_metric = {
    "algorithm": "",
    "scenario": "",
    "is_safe": "Safe success rate ${}$".format(METRIC_BOOL),
    "is_not_safe": METRIC_BOOL,
    "is_safe_and_success": "Success rate",
    "is_unsafe_and_success": METRIC_BOOL,
    "goal_success": METRIC_BOOL,
    "is_touching_speed": METRIC_SPEED,
    "is_hitting_speed": METRIC_SPEED,
    "is_hitting_distance": METRIC_DISTANCE,
    "is_touching_distance": METRIC_DISTANCE,
    "distance_to_nearest_obstacle": "Minimum Clearance {}".format(METRIC_DISTANCE),
    "distance_to_goal": "Distance to goal {}".format(METRIC_DISTANCE),
    "avg_local_time_step": METRIC_TIME,
    "avg_global_time_step": METRIC_TIME,
    "total_planning_and_exec_time": METRIC_TIME,
    "smoothness": METRIC_SMOOTHNESS,
    "total_joints_effort": "Trajectory length {}".format(METRIC_SMOOTHNESS),
}
y_axis_range = {
    "is_safe_and_success": [0, 1],
    "distance_to_nearest_obstacle": [0, 0.3],
    "distance_to_goal": [0, 0.3],
    "total_joints_effort": [0, 7],
}
# %%

# algs = ["gsd","gtt", "lsd",  "gsm", "ggd", "mod", "moS", "msd", "gdg", "gln"]
# algs_names = {"gsd": "GLIR smooth", "lsd": "Local", "ggd": "Global", "gsm": "machines", "mod": "rrtconnect",
#               "moS": "lazyprm", "msd": "stomp", "gdg": "GLIR dynamic", "gln": "GLIR not smooth", }
# algs = ["ggd", "gs3", "gs2", "gs1", "gs0", "ls0", "mod", "moS", "msd"]
# algs = ["ggd", "gs3", "gs1", "gs0", "ls0"]


# Naming for the scatter plot
algs_names = {
    "ggd": "Global",
    'glir_0.01': "0.01",
    'glir_0.1': "0.1",
    'glir_0.2': "0.2",
    'glir_0.3': "0.3",
    'glir_0.4': "0.4",
    'glir_0.5': "0.5",
    'glir_1.0': "1",
    'glir_10.0': "10",
    # 'glir_s100': "100",
    "mod": "RRTConnect*",
    "moS": "SemiPersistentLazyPRM*",
    "moP": "PRM*",
    "moL": "LazyPRM*",
    "msd": "STOMP"
    # "ls1": "Local multi-goal",
}
# Naming for the box plot
# algs_names = {
#     'glir_s.4': "GLIR (ours)",
#     "mod": "RRTConnect*",
#     "moS": "LazyPRM*",
#     "msd": "STOMP"
# }
# %%
# title_text = '$\LARGE\eta_p$'
title_text = ''

# TODO currently show graphs ONLY when a trajectory is executed

colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1
# x and y-axis formatting
font_dict = dict(family='Arial',
                 size=20,
                 color='black'
                 )


# plot_columns = [sub_df.is_safe, sub_df.total_joints_effort]


# plot_columns = [sub_df.distance_to_nearest_obstacle, sub_df.distance_to_goal]
# plot_columns = [sub_df.is_safe_and_success, sub_df.total_joints_effort]


# for ic, c in enumerate(sub_df.columns[2:]):

def plot(exp_dir=None):
    fig_bp = make_subplots(
        rows=len(plot_columns), cols=len(scenarios),
        subplot_titles=[scenario_names.get(scenario_) for scenario_ in scenarios],
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
        shared_xaxes=True,
        shared_yaxes=True,
    )
    for ir, sub_df_c in enumerate(plot_columns):
        for iscene, scen in enumerate(scenarios):
            if ir == len(plot_columns) - 1 and iscene == len(scenarios) - 1:
                show_legend = True
            else:
                show_legend = False

            if iscene == len(scenarios) - 1:
                fig_bp.update_yaxes(title_text=y_axis_metric.get(sub_df_c.name, sub_df_c.name), row=ir + 1, col=1)

            for ialg, alg in enumerate(algs):
                y = sub_df_c[(sub_df.algorithm == alg) &
                             (sub_df.scenario == scen) #&
                             # (sub_df.is_safe_and_success == 1.0)
                             # (sub_df.n_timestep > 2.0)
                             ].to_numpy()  # alg, scene --> c
                fig_bp.add_trace(Box(
                    y=y,
                    # boxmean="sd",  # represent mean
                    name=algs_names.get(alg, alg),  # '{}_{}'.format(alg, c),
                    boxpoints='all',
                    legendgroup=alg,  # '{}_{}'.format(alg, c),
                    showlegend=show_legend,
                    hovertemplate='<b>%{text}</b>: <i>%{x}</i>: %{y:.2f}',
                    text=[alg] * len(y),
                    marker_color=colors[ialg],
                    marker_size=2,

                ),
                    row=ir + 1,
                    col=iscene + 1,
                )

    # =========

    fig_bp.update_yaxes(
        # title_text='Y-axis',  # axis label
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
    fig_bp.update_yaxes(title_standoff=0, row=1, col=1)
    fig_bp.update_yaxes(title_standoff=25, row=2, col=1)

    fig_bp.update_xaxes(
        title_text=title_text,
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
    # =========
    if len(algs) <= 5:
        x = 0.98
    else:
        x = 0.9
    fig_bp.update_layout(template="simple_white",
                         legend=dict(
                             yanchor="top",
                             y=0.48,
                             xanchor="right",
                             x=x,
                             bordercolor="Black",
                             borderwidth=1,
                             bgcolor="rgba(255,255,255,0)",
                             font=dict(family='Arial',
                                       size=18,
                                       color='black'
                                       )
                         ),
                         font=font_dict
                         )
    fig_bp.update_annotations(
        font=font_dict,
        y=1.01
    )
    if exp_dir:
        image_name = join(exp_dir, "fig1.pdf")
    else:
        image_name = "fig1.pdf"
    fig_bp.write_image(image_name, width=2311, height=1202, scale=1)
    time.sleep(2)
    fig_bp.write_image(image_name, width=2311, height=1202, scale=1)
    fig_bp.show()


def plot_compact_alg(exp_dir=None, scatter_or_box_plot="scatter"):
    fig_bp = make_subplots(
        rows=1, cols=len(plot_columns),
        # subplot_titles=[scenario_names.get(scenario_) for scenario_ in scenarios],
        horizontal_spacing=0.08,
        # vertical_spacing=0.01,
        # shared_xaxes=True,
        # shared_yaxes=True,
    )
    for ir, sub_df_c in enumerate(tqdm.tqdm(plot_columns)):
        # for iscene, scen in enumerate(scenarios):
        if ir == len(plot_columns) - 1:
            show_legend = True
        else:
            show_legend = False

        # if iscene == len(scenarios) - 1:
        fig_bp.update_yaxes(title_text=y_axis_metric.get(sub_df_c.name, sub_df_c.name), row=1, col=ir + 1)

        means = []
        errors = []
        if scatter_or_box_plot == "scatter":
            for ialg, alg in enumerate(algs):
                y = sub_df_c[(sub_df.algorithm == alg)
                             # & (sub_df.n_timestep > 2.0)
                             ].to_numpy()  # alg, scene --> c
                means.append(np.mean(y))
                errors.append(np.std(y))
            data = Scatter(
                x=[algs_names.get(alg, alg) for alg in algs],
                y=means,
                error_y=dict(
                    type='data',  # value of error bar given in data coordinates
                    array=np.array(errors) / 2,
                    visible=True),
                name=algs_names.get(alg, alg),  # '{}_{}'.format(alg, c),
                # boxpoints='all',
                legendgroup=alg,  # '{}_{}'.format(alg, c),
                showlegend=False,
                hovertemplate='<b>%{text}</b>: <i>%{x}</i>: %{y:.2f}',
            )
            fig_bp.add_trace(data, row=1, col=ir + 1, )

        else:
            print("*" * 50)
            print("Evaluating: ", sub_df_c.name)
            print("*" * 50)
            for ialg, alg in enumerate(algs):
                # alg_all_scenarios_y = sub_df_c[(sub_df.algorithm == alg) &
                #                                (sub_df.n_timestep > 2.0)].to_numpy()
                alg_all_scenarios_y = sub_df_c[(sub_df.algorithm == alg)].to_numpy()
                mean = np.nanmean(alg_all_scenarios_y)
                std = np.nanstd(alg_all_scenarios_y)
                print("{} : {:.2f} +- {:.2f} ".format(algs_names.get(alg, alg), mean, std))
                # print("{} : {:.2%} +- {:.2%} ".format(algs_names.get(alg, alg), mean, std))
                for scen in scenarios:
                    y = sub_df_c[(sub_df.algorithm == alg) &
                                 # (sub_df.n_timestep > 2.0) &
                                 (sub_df.scenario == scen)
                                 ].to_numpy()  # alg, scene --> c
                    means.append(np.nanmean(y))
                    errors.append(np.nanstd(y))
                print("{} : {:.2f} +- {:.2f} ".format(algs_names.get(alg, alg), np.nanmean(means), np.nanstd(means)))
                # print("{} : {:.2%} +- {:.2%} ".format(algs_names.get(alg, alg), np.mean(means), np.std(means)))

                data = Box(
                    # x=[algs_names.get(alg, alg) for alg in algs],
                    y=means,
                    boxmean="sd",  # represent mean
                    name=algs_names.get(alg, alg),  # '{}_{}'.format(alg, c),
                    # boxpoints='all',
                    legendgroup=alg,  # '{}_{}'.format(alg, c),
                    showlegend=False,
                    hovertemplate='<b>%{text}</b>: <i>%{x}</i>: %{y:.2f}',
                    text=[alg] * len(y),
                    marker_color=colors[ialg],
                    marker_size=2, )
                fig_bp.add_trace(data, row=1, col=ir + 1, )

            # data = Scatter(
            #     x=[algs_names.get(alg, alg) for alg in algs],
            #     y=means,
            #     error_y=dict(
            #         type='data',  # value of error bar given in data coordinates
            #         array=np.array(errors) / 2,
            #         visible=True),
            #     name=algs_names.get(alg, alg),  # '{}_{}'.format(alg, c),
            #     # boxpoints='all',
            #     legendgroup=alg,  # '{}_{}'.format(alg, c),
            #     showlegend=False,
            #     hovertemplate='<b>%{text}</b>: <i>%{x}</i>: %{y:.2f}',
            # )
        # y_axis_range_ = y_axis_range.get(sub_df_c.name)
        # if y_axis_range_ is not None:
        #     fig_bp.update_layout(yaxis_range=y_axis_range_, overwrite=False)

    # =========

    fig_bp.update_yaxes(
        # title_text='Y-axis',  # axis label
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
    fig_bp.update_yaxes(title_standoff=4, row=1, col=1, range=y_axis_range.get(plot_columns[1 - 1].name))
    fig_bp.update_yaxes(title_standoff=0, row=1, col=2, range=y_axis_range.get(plot_columns[2 - 1].name))
    fig_bp.update_yaxes(title_standoff=0, row=1, col=3, range=y_axis_range.get(plot_columns[3 - 1].name))
    fig_bp.update_yaxes(title_standoff=0, row=1, col=4, range=y_axis_range.get(plot_columns[4 - 1].name))

    fig_bp.update_xaxes(
        title_text=title_text,
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
    # =========
    if len(algs) <= 5:
        x = 0.98
    else:
        x = 0.9
    fig_bp.update_layout(template="simple_white",
                         legend=dict(
                             # yanchor="top",
                             # y=0.48,
                             # xanchor="right",
                             # x=x,
                             bordercolor="Black",
                             borderwidth=1,
                             bgcolor="rgba(255,255,255,0)",
                             font=dict(family='Arial',
                                       size=18,
                                       color='black'
                                       )
                         ),
                         font=font_dict
                         )
    # fig_bp.update_annotations(
    #     font=font_dict,
    #     y=1.01
    # )
    if exp_dir:
        image_name = join(exp_dir, "fig1.pdf")
    else:
        image_name = "fig1.pdf"
    try:
        fig_bp.write_image(image_name, width=2311, height=600, scale=1)
    except:
        traceback.print_exc()
    time.sleep(2)
    try:
        fig_bp.write_image(image_name, width=2311, height=600, scale=1)
    except:
        traceback.print_exc()

    fig_bp.show()


def plot_compact_alg_per_scenario(exp_dir=None):
    fig_bp = make_subplots(
        rows=1, cols=len(plot_columns),
        # subplot_titles=[scenario_names.get(scenario_) for scenario_ in scenarios],
        # horizontal_spacing=0.01,
        # vertical_spacing=0.01,
        # shared_xaxes=True,
        # shared_yaxes=True,
    )
    for ir, sub_df_c in enumerate(plot_columns):
        for iscene, scen in enumerate(scenarios):
            if ir == len(plot_columns) - 1:
                show_legend = True
            else:
                show_legend = False

            # if iscene == len(scenarios) - 1:
            fig_bp.update_yaxes(title_text=y_axis_metric.get(sub_df_c.name, sub_df_c.name), row=1, col=ir + 1)

            means = []
            errors = []
            for ialg, alg in enumerate(algs):
                y = sub_df_c[(sub_df.algorithm == alg) &
                             (sub_df.scenario == scen) &
                             (sub_df.n_timestep > 2.0)
                             ].to_numpy()  # alg, scene --> c
                means.append(np.mean(y))
                errors.append(np.std(y))
            fig_bp.add_trace(Scatter(
                x=[algs_names.get(alg, alg) for alg in algs],
                y=means,
                error_y=dict(
                    type='data',  # value of error bar given in data coordinates
                    array=np.array(errors) / 2,
                    visible=True),
                name=algs_names.get(alg, alg),  # '{}_{}'.format(alg, c),
                # boxpoints='all',
                legendgroup=alg,  # '{}_{}'.format(alg, c),
                showlegend=show_legend,
                hovertemplate='<b>%{text}</b>: <i>%{x}</i>: %{y:.2f}',
            ),
                row=1,
                col=ir + 1,
            )
    # =========

    fig_bp.update_yaxes(
        # title_text='Y-axis',  # axis label
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
    fig_bp.update_yaxes(title_standoff=0, row=1, col=1)
    fig_bp.update_yaxes(title_standoff=0, row=1, col=2)
    fig_bp.update_yaxes(title_standoff=0, row=1, col=3)

    fig_bp.update_xaxes(
        title_text=title_text,
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
    # =========
    if len(algs) <= 5:
        x = 0.98
    else:
        x = 0.9
    fig_bp.update_layout(template="simple_white",
                         legend=dict(
                             # yanchor="top",
                             # y=0.48,
                             # xanchor="right",
                             # x=x,
                             bordercolor="Black",
                             borderwidth=1,
                             bgcolor="rgba(255,255,255,0)",
                             font=dict(family='Arial',
                                       size=18,
                                       color='black'
                                       )
                         ),
                         font=font_dict
                         )
    # fig_bp.update_annotations(
    #     font=font_dict,
    #     y=1.01
    # )
    if exp_dir:
        image_name = join(exp_dir, "fig1.pdf")
    else:
        image_name = "fig1.pdf"
    fig_bp.write_image(image_name, width=2311, height=800, scale=1)
    time.sleep(2)
    fig_bp.write_image(image_name, width=2311, height=800, scale=1)

    fig_bp.show()


# %%
# Total planning + execution time
# Total trajectory time Total planning + execution time
# Trajectory length

# ===========================================
# Calc table values paper
# ===========================================
#     fig_bp = Figure()
#     algs = set(list(sub_df.algorithm))
#     for c in sub_df.columns[1:]:
#         fig_bp = Figure()
#         for ialg, alg in enumerate(algs):
#             for iscene, scen in enumerate(algs):
#                 show_legend = True if ialg == len(algs) - 1 else False
#                 y = sub_df[c][sub_df.algorithm == alg].to_numpy()
#                 x = [iscene] * len(y)
#                 fig_bp.add_trace(Box(
#                     x=x,
#                     y=y,
#                     boxmean="sd",  # represent mean
#                     name=ialg + iscene,  # '{}_{}'.format(alg, c),
#                     boxpoints='all',
#                     legendgroup=c,  # '{}_{}'.format(alg, c),
#                     showlegend=show_legend,
#                     hovertemplate=
#                     '<i>%{x}</i>: $%{y:.2f}' +
#                     '<b>%{text}</b>',
#                     text=[alg] * len(y),
#                 )
#                 )
if __name__ == '__main__':
    # exp_name = "/home/mohamed/git/free_space/experiments/glir/Results/analysis/gen_exp_dataframes.csv"
    # exp_name = "/home/mohamed/git/free_space/experiments/glir/Results/analysis_2/gen_exp_dataframes.csv"
    # exp_name = "/home/mohamed/git/free_space/experiments/glir/Results_corrected_distance/glir_vrep_0.25/gen_exp_dataframes.csv"
    # exp_name = "/home/mohamed/git/free_space/experiments/glir/Results_corrected_distance/analysis/gen_exp_dataframes.csv"
    # exp_name = "/home/mohamed/git/glir/experiments/Results_analysis/corrected_octomap_5cm/moveit/gen_exp_dataframes.csv"
    exp_name = "/home/mohamed/git/glir/experiments/Results_analysis/corrected_octomap_0.025_col/moveit/gen_exp_dataframes.csv"
    # exp_name = "/home/mohamed/git/glir/experiments/Results_analysis/corrected_octomap_2.5cm/moveit/gen_exp_dataframes.csv"
    exp_name = r"C:\Users\mae\git\demonstrator\phd_experiments\static_exp\glir\glir\gen_exp_dataframes.csv"
    exp_dirs = [
        r"C:\Users\mae\git\demonstrator\phd_experiments\static_exp\moveit\2304_moveit_updated",
        r"C:\Users\mae\git\demonstrator\phd_experiments\static_exp\neo\neo",
        r"C:\Users\mae\git\demonstrator\phd_experiments\static_exp\glir\glir",
        r"C:\Users\mae\git\demonstrator\phd_experiments\static_exp\machines\machines",
    ]
    sub_df = None
    for exp_dir_ in exp_dirs:
        exp_name = join(exp_dir_, "gen_exp_dataframes.csv")
        sub_df_ = pd.read_csv(exp_name, index_col=0)
        if sub_df is None:
            sub_df = sub_df_
        else:
            sub_df = pd.concat([sub_df, sub_df_])

    algs = set(list(sub_df.algorithm))
    # algs = [
    #     "glir_neo",
    #     "glir",
    # ]
    #
    # "avg_local_time_step",
    #     # "avg_global_time_step",
    print(algs)
    plot_columns = [
        sub_df.is_safe_and_success,
        sub_df.distance_to_nearest_obstacle,
        sub_df.distance_to_goal,
        # sub_df.is_not_safe,
        # sub_df.avg_local_time_step,
        # sub_df.avg_global_time_step,
        sub_df.total_joints_effort

    ]
    fig = Figure()
    save_exp_dir = r"C:\Users\mae\git\demonstrator\phd_experiments\static_exp\analyse"
    # plot(exp_dir=exp_dir)
    # plot_compact_alg(exp_dir=save_exp_dir, scatter_or_box_plot="scatter")
    plot_compact_alg(exp_dir=save_exp_dir, scatter_or_box_plot="box")

    # plot_compact_alg_per_scenario(exp_dir=exp_dir)

# x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# y_axis_range = [
#
# ]
# TODO: create a pandas frame with the results
results_columns = [
    "is_safe_and_success",
    "total_joints_effort",
    "distance_to_goal",
    "smoothness",
    "is_not_safe",
    "avg_local_time_step",
    "avg_global_time_step",
    "total_planning_and_exec_time",
]
groupby_alg = sub_df.groupby("algorithm")
print(groupby_alg.mean())
print(groupby_alg.std())

results_as_table = pd.DataFrame(columns=results_columns)
for p in results_columns:
    # TODO: success rate and collisions mean and std should be considered. The rest: mean and std of the whole data frame
    y = []
    errors = []
    for alg in algs:
        msd = [sub_df[p][(sub_df.algorithm == alg) & (sub_df.scenario == scen)].to_numpy() for scen in
               scenarios]
        msd_means = np.array([np.nanmean(msd_) for msd_ in msd])
        mean = np.nanmean(msd_means)
        std = np.nanstd(msd_means)
        print(algs_names.get(alg, alg), p, mean, std)
        y.append(mean)
        errors.append(std)
print(errors)
# TODO check this link to auto generate latex tables: https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.to_latex.html#pandas.io.formats.style.Styler.to_latex

#     y = np.array(y)
#     errors = np.array(errors)
#     y_upper = list(y + errors)
#     y_lower = list(y - errors)
#     y = list(y)
#     # fig.add_trace(
#     #     go.Scatter(
#     #         x=x,
#     #         y=y,
#     #         line=dict(color='rgb(0,100,80)'),
#     #         mode='lines'
#     #     ),
#     # )
#     # fig.add_trace(
#     #     go.Scatter(
#     #         x=x  + x[::-1],  # x, then x reversed
#     #         y=y_upper + y_lower[::-1],  # upper, then lower reversed
#     #         fill='toself',
#     #         fillcolor='rgba(0,100,80,0.2)',
#     #         line=dict(color='rgba(255,255,255,0)'),
#     #         hoverinfo="skip",
#     #         showlegend=False
#     #     ))
#     fig.add_trace(go.Scatter(
#         x=x,
#         y=y,
#         error_y=dict(
#             type='data',  # value of error bar given in data coordinates
#             array=errors,
#             visible=True)
#     ))
#     print("*" * 50)
#
# fig.update_layout(yaxis_range=[0, 1])
#
# fig.show()
# # TODO plot means and variances: smoothness vs different metrics
# # print("Statistics finished")
# # Global is_safe_and_success 0.002717391304347826 0.0047066598031762965
# # Global total_joints_effort 5.2360053616310855 0.21336628442428
# # Global distance_to_goal 0.05511732690094885 0.0020147631411913005
# # Global smoothness nan nan
# # **************************************************
# # GLIR not smooth is_safe_and_success 0.6069334650856391 0.07563055484487384
# # GLIR not smooth total_joints_effort 7.186954598232176 0.20780447821604683
# # GLIR not smooth distance_to_goal 0.02815418221304946 0.00754936805637313
# # GLIR not smooth smoothness 3.138748818561655 0.00014052074175617565
# # **************************************************
# # GLIR smooth is_safe_and_success 0.5475 0.07361215932167728
# # GLIR smooth total_joints_effort 2.7566335021195667 0.1458408407855172
# # GLIR smooth distance_to_goal 0.031033004614402823 0.011116847089335361
# # /home/mohamed/git/free_space/robot_control/planner/exp_generator/analysis/statistics_paper_plots.py:226: RuntimeWarning: Mean of empty slice
# #   msd_means = np.array([np.nanmean(msd_) for msd_ in msd])
# # GLIR smooth smoothness 2.996984554975689 0.09254592506260119
# # **************************************************
# # GLIR very smooth is_safe_and_success 0.55 0.06819090848492927
# # GLIR very smooth total_joints_effort 2.7811207813764174 0.12290838080143127
# # GLIR very smooth distance_to_goal 0.02874899376903636 0.009349819229111569
# # GLIR very smooth smoothness 2.997104773650984 0.09265209606749028
# # **************************************************
# # Local is_safe_and_success 0.5900000000000001 0.103440804327886
# # Local total_joints_effort 3.020877650369434 0.02972125765925303
# # Local distance_to_goal 0.03389068128388645 0.010573785617780995
# # Local smoothness 3.1387372861225193 0.00047216718013801286
# # **************************************************
# #
#
# # =====================================================================================================================
# #                           New results: 09.04.2022
# # =====================================================================================================================
# # Local is_safe_and_success 0.11750000000000001 0.05717298313014636
# # GLIR smooth is_safe_and_success 0.0975 0.05889609494694874
# # GLIR smooth is_safe_and_success 0.0975 0.05889609494694874
# # GLIR very smooth is_safe_and_success 0.145 0.06689544080129826
# # GLIR not smooth is_safe_and_success 0.030000000000000002 0.03082207001484489
