# %%
"""
Compute statistics saved in RobotControlMetric dataholder
1. access to all monitor.pb files in exp_dir
2. for each 10 episodes: calculate the statistics
3. calculate mean and variance between the different chunks
4. Visualize the metrics
"""
import os

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import time

from experiments.analysis.statistics_static_env_paper_plots import y_axis_metric, font_dict, colors

import os.path
from os.path import join
from plotly.graph_objs import Scatter

import pandas as pd
from plotly.subplots import make_subplots

pd.options.plotting.backend = "plotly"

from cara.workplaces.obstacles_workplace import ObstacleWorkplaceEnv

scenarios = [ObstacleWorkplaceEnv.SCENARIO_BINS,
             ObstacleWorkplaceEnv.SCENARIO_REAL_OBJECTS,
             ObstacleWorkplaceEnv.SCENARIO_REGAL,
             ObstacleWorkplaceEnv.SCENARIO_PRIMITIVE_OBJECTS
             ]


# %%
# exp_dir = "/home/mohamed/git/free_space/robot_control/planner/exp_generator/experiments/Results/dynamic_environment"
# exp_dir = "/home/mohamed/git/free_space/robot_control/planner/video/Results"
# exp_dir = "/home/mohamed/git/free_space/robot_control/planner/video/analysis"
# df_all = all_to_df(exp_dir)
# df_all.to_csv(join(exp_dir, "dynamic_pd.csv"))


def plot_compact_alg_per_scenario(algs, algs_names, exp_dir=None):
    fig_bp = make_subplots(
        rows=1, cols=len(plot_columns),
        # subplot_titles=[scenario_names.get(scenario_) for scenario_ in scenarios],
        horizontal_spacing=0.15,
        # vertical_spacing=0.01,
        # shared_xaxes=True,
        # shared_yaxes=True,
    )
    for ir, sub_df_c in enumerate(plot_columns):
        for ialg, alg in enumerate(algs):
            if ir == len(plot_columns) - 1:
                show_legend = True
            else:
                show_legend = False

            # if iscene == len(scenarios) - 1:
            fig_bp.update_yaxes(title_text=y_axis_metric.get(sub_df_c.name, sub_df_c.name), row=1, col=ir + 1)

            fig_bp.add_trace(Scatter(
                y=sub_df_c[(sub_df.algorithm == alg)],
                showlegend=show_legend,
                hovertemplate='<b>%{text}</b>: <i>%{x}</i>: %{y:.2f}',
                name=algs_names.get(alg, alg),  # '{}_{}'.format(alg, c),
                legendgroup=alg,  # '{}_{}'.format(alg, c),
                marker_color=colors[ialg],
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

    fig_bp.update_xaxes(
        title_text='timestep',
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
                             y=0.93,
                             xanchor="right",
                             x=0.73,
                             bordercolor="Black",
                             borderwidth=1,
                             bgcolor="rgba(255,255,255,0)",
                             # font=dict(family='Arial',
                             #           size=24,
                             #           color='black'
                             #           )
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
    fig_bp.write_image(image_name, width=2311 // 2, height=600, scale=1)
    time.sleep(2)
    # write it again otherwise we have a wierd print in the pdf output from plotly (bug)
    fig_bp.write_image(image_name, width=2311 // 2, height=600, scale=1)
    fig_bp.show()


if __name__ == '__main__':
    df_name = r"C:\Users\mae\git\demonstrator\phd_experiments\static_obstacle_dynamic_goal_exp\dynamic_goal\gen_exp_dataframes.csv"
    df_name = r"C:\Users\mae\git\demonstrator\phd_experiments\static_obstacle_dynamic_goal_exp\dynamic_goal_hard\gen_exp_dataframes.csv"
    sub_df = pd.read_csv(df_name, index_col=0)
    plot_columns = [
        sub_df.distance_to_nearest_obstacle,
        sub_df.distance_to_goal,
    ]
    algs = sub_df["algorithm"].unique()
    # algs_names = {k:k for k in algs}
    algs_names = {"glir": "GLIR",
                  "glir_local": "GLIR LP",
                  "machines": "PFM",
                  "neo": "Neo",
                  }
    plot_compact_alg_per_scenario(algs, algs_names, exp_dir=os.path.dirname(df_name))
