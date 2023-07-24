import json
import os
import pickle

import pandas as pd
from Evaluation.plotly_setup import font_dict, title_text, colors
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_post_hoc_datatsets():
    df = pd.read_csv('../data/datasets.csv')
    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=3)
    y = df['mean'].values
    x1 = df['after_nr_features'].values
    x2 = df['nr_feature'].values

    fig.add_trace(
        go.Scatter(x=x2, y=y, mode='markers', name='Nr features before FE', marker_color=colors[0]),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=x1, y=y, mode='markers', name='Nr Features after FE ', marker_color=colors[1]),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x1 - x2, y=y, mode='markers', name='Difference',
                   hovertemplate=
                   '<br>Avg performance: %{y}<br>' +
                   '(before,after): %{text}',
                   text=[f'({before},{after})' for after, before in zip(x1, x2)], marker_color=colors[2]),
        row=1, col=3
    )

    # fig.add_trace(
    #     go.Scatter(x=x1 + x2, y=y, mode='markers', name='samples+features'),
    #     row=2, col=2
    # )
    # Change the bar mode
    fig.update_yaxes(
        title_text='Avg (After-Before)FE Accuracy (%)',  # axis label
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
        title_text='Nr of  Features',
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
    fig.update_layout(barmode='group',
                      title='Number (Nr) of Featuers vs Predictive Performance',
                      template="simple_white",
                      legend=dict(
                          yanchor="top",
                          y=0.9,
                          xanchor="right",
                          x=0.9,
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
    image_name = 'nr_features_vs_acc'
    fig.write_html(f'{image_name}.html')
    fig.write_image(f'paper/{image_name}.pdf', width=2311, height=1202, scale=1)
    time.sleep(2)
    fig.write_image(f'paper/{image_name}.pdf', width=2311, height=1202, scale=1)
    fig.show()


def plot_performance_gain():
    global name
    directory = 'wandb_data'
    filename = 'opt_cfg.csv'
    csv_path = os.path.join(directory, filename)
    df = pd.read_csv(csv_path)
    df["test_diff"] = df['final_test_score'] * 100 - df["first_test_score"] * 100
    df["train_diff"] = df['final_best_fitness'] * 100 - df["first_best_fitness"] * 100
    result = df.groupby(['Name'], as_index=False).agg(
        {'test_diff': ['mean', 'std'], 'train_diff': ['mean', 'std']})
    result = result.sort_values(('train_diff', 'mean'))
    names = [str(name) for name in result['Name'].values[:-2]]
    df_test = result['test_diff'][:-2]
    df_train = result['train_diff'][:-2]
    fig = go.Figure(data=[
        go.Bar(name='Test', x=names, y=df_test['mean'], error_y=dict(type='data', array=df_test['std']),
               marker_color=colors[4]),
        go.Bar(name='Valid', x=names, y=df_train['mean'], error_y=dict(type='data', array=df_train['std']),
               marker_color=colors[5])

    ])
    # Change the bar mode
    fig.update_yaxes(
        title_text='Avg (After-Before)FE Accuracy (%)',  # axis label
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
        title_text='Dataset ID',
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
    fig.update_layout(barmode='group',
                      title='Average Predictive Performance Change per Dataset',
                      template="simple_white",
                      legend=dict(
                          yanchor="top",
                          y=0.9,
                          xanchor="right",
                          x=0.9,
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
    image_name = 'diff_bar_paper'
    fig.write_html(f'{image_name}.html')
    fig.write_image(f'paper/{image_name}.pdf', width=2311, height=1202, scale=1)
    time.sleep(2)
    fig.write_image(f'paper/{image_name}.pdf', width=2311, height=1202, scale=1)
    fig.show()


def get_opt_config_data():
    directory = '../results_bo'
    datasets = [458, 15, 1068, 11, 22, 16, 1510, 18, 31, 469, 188, 37, 54, 14, 23, 50]
    rows = []
    for dataset in datasets:
        opt_config = os.path.join(directory, str(dataset), 'opt_cfg.json')
        with open(opt_config) as json_file:
            data = json.load(json_file)
        data['id'] = dataset
        rows.append(data)

    df = pd.DataFrame(rows)
    df.to_csv('opt_configurations_found.csv')


def plot_post_hoc_bo():
    data = pd.read_csv('opt_configurations_found.csv')
    categorical_cols = ['children_per_step', 'max_pop_size',
                        'population_size', 'selection_type',
                        'total_number_of_function_evaluations']
    fig = make_subplots(rows=1, cols=len(categorical_cols), shared_yaxes=True, horizontal_spacing=0.01)
    for icol, col in enumerate(categorical_cols):
        group = data.groupby([col])[col].count()
        values = group.index.values
        x = [str(val) for val in values]
        y = [group[val] for val in values]
        fig.add_trace(go.Bar(x=x, y=y, showlegend=False), row=1, col=icol + 1)
        fig.update_xaxes(
            title_text=col,
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
    fig.update_layout(xaxis_type='category', bargap=0, bargroupgap=0)
    # Change the bar mode
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

    fig.update_layout(barmode='group',
                      title='Optimal Configurations',
                      template="simple_white",
                      legend=dict(
                          yanchor="top",
                          y=0.9,
                          xanchor="right",
                          x=0.9,
                          bordercolor="Black",
                          borderwidth=1,
                          bgcolor="rgba(255,255,255,0)",
                          font=dict(family='Arial',
                                    size=20,
                                    color='black'
                                    )
                      ),
                      font=font_dict,
                      xaxis_title = 'Hyperparameters',
                      )
    fig.write_html('opt_configs.html')

    #     if col == 'id':
    #         continue
    #     fig.add_trace(go.Box(y=data[col]))
    # fig.write_image('opt_configs.html')


if __name__ == '__main__':
    #get_opt_config_data()
    plot_post_hoc_bo()
    plot_post_hoc_datatsets()
    # plot_performance_gain()
