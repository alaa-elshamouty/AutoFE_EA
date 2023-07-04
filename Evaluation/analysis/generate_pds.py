# %%
"""
Compute statistics saved in RobotControlMetric dataholder
1. access to all monitor.pb files in exp_dir
2. for each 10 episodes: calculate the statistics
3. calculate mean and variance between the different chunks
4. Visualize the metrics
"""
import os
import traceback

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import glob
import os.path
from os.path import join

import numpy as np
import pandas as pd

from tqdm import tqdm

from cara.robosuite.robosuite.models.robots.utils import load_object
from cara.workplaces.obstacles_workplace import ObstacleWorkplaceEnv

scenarios = [ObstacleWorkplaceEnv.SCENARIO_BINS,
             ObstacleWorkplaceEnv.SCENARIO_REAL_OBJECTS,
             ObstacleWorkplaceEnv.SCENARIO_REGAL,
             ObstacleWorkplaceEnv.SCENARIO_PRIMITIVE_OBJECTS
             ]


def scenario_to_df(episodes, exp_dir=None, scenario_=None, load_pd_if_Exists=False, exp_name="", dt=None,
                   goal_tol=3e-2):
    if exp_dir is not None and scenario_ is not None:
        df_path = join(exp_dir, "pd_" + scenario_ + ".csv")
    else:
        df_path = None

    if dt is None:
        # extract dt from exp_config.pb
        exp_config_path = join(exp_dir, "exp_config.pb")
        exp_config = load_object(path=exp_config_path)
        dt = exp_config.env_config.get("dt")
    if load_pd_if_Exists and df_path is not None and os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        episodes_stats_keys = ["goal_success",
                               "is_safe",
                               "is_safe_and_success",
                               "is_hitting_distance",
                               "is_hitting_speed",
                               "is_touching_distance",
                               "is_touching_speed",
                               "total_global_time",
                               "total_local_time_step",
                               "tcp_max_speed",
                               "avg_local_time_step",
                               "avg_global_time_step",
                               "distance_to_goal",
                               "distance_to_nearest_obstacle",
                               "total_joints_effort",
                               "n_timestep",
                               "smoothness",
                               "lp_infeasible"
                               ]
        df = pd.DataFrame(columns=episodes_stats_keys)
        for iepisode, e in enumerate(tqdm(episodes)):
            # e.plot_properties(save_dir=output_dir, n_episode=iepisode)
            # e.plot_properties(n_episode=iepisode)
            try:
                if len(e.distance_to_nearest_obstacle) == 0:
                    raise ValueError("Empty distance_to_nearest_obstacle")
                goal_success = e.is_success(tol=3e-2)  # cm
                is_safe = e.is_safe(0.)
                is_not_safe = 1 - is_safe
                is_safe_and_success = e.is_safe_and_success(tol_safe=0., tol_success=goal_tol)  # 3cm
                is_unsafe_and_success = e.is_unsafe_and_success()
                distance_hit, tcp_speed_hit = e.is_not_safe(-1e-2)
                distance_touch, tcp_speed_touch = e.is_touching(-5e-2)  # 5 cm
                try:
                    geometric_time = np.sum(e.gp_geometric_time)
                    total_global_query_time = np.sum(e.global_planner_total_query_time)
                    avg_global_time_step = np.mean(e.global_planner_step_time)
                    total_planning_and_exec_time = np.sum(e.local_planner_step_time) + np.sum(
                        e.global_planner_step_time)

                except:
                    geometric_time = 0.
                    total_global_query_time = 0.
                    avg_global_time_step = 0.
                    total_planning_and_exec_time = 0.
                if total_planning_and_exec_time == 0. and dt is not None:
                    total_planning_and_exec_time = dt * 1000. * e.local_planner_substep
                total_local_time_step = np.sum(e.local_planner_step_time) or np.sum(e.step_time)
                avg_local_time_step = np.mean(e.local_planner_step_time)
                avg_local_time_step = np.mean(e.step_time) if np.isnan(
                    np.mean(e.local_planner_step_time)) else avg_local_time_step
                distance_to_goal = np.min(e.distance_to_goal)
                distance_to_nearest_obstacle = np.min(e.distance_to_nearest_obstacle)
                total_joints_effort = e.total_joints_effort
                n_timestep = e.n_timestep
                smoothness = e.smoothness
                try:
                    manipulability = np.mean(e.manipulability)
                except:
                    manipulability = 0.
                total_execution_time = e.total_execution_time
                tcp_max_speed = np.max(e.tcp_speed)
                try:
                    lp_infeasible = int(np.max(e.lp_infeasible))
                except:
                    lp_infeasible = -1

            except ValueError:
                goal_success = 0.
                is_safe = 1.
                is_not_safe = 1 - is_safe
                is_safe_and_success = 0.
                is_unsafe_and_success = 0.
                distance_hit, tcp_speed_hit = 0., 0.
                distance_touch, tcp_speed_touch = 0., 0.,
                geometric_time = np.sum(e.gp_geometric_time)
                total_global_query_time = np.sum(e.global_planner_total_query_time)
                total_local_time_step = 0.
                avg_local_time_step = 0.
                avg_global_time_step = 0.
                total_planning_and_exec_time = 0.
                tcp_max_speed = 0.
                distance_to_goal = np.nan
                distance_to_nearest_obstacle = np.nan
                total_joints_effort = 0.
                n_timestep = 0.
                smoothness = 0.
                total_execution_time = 0.
                lp_infeasible = -1
                manipulability = 0.

            except Exception:
                traceback.print_exc()
            episodes_stats = {
                "goal_success": goal_success,
                "is_safe": is_safe,
                "is_not_safe": is_not_safe,
                "is_safe_and_success": is_safe_and_success,
                "is_unsafe_and_success": is_unsafe_and_success,
                "is_hitting_distance": distance_hit,
                "is_hitting_speed": tcp_speed_hit,
                "is_touching_distance": distance_touch,
                "is_touching_speed": tcp_speed_touch,

                "geometric_time": geometric_time,
                "total_global_query_time": total_global_query_time,

                "total_local_time_step": total_local_time_step,
                "avg_local_time_step": avg_local_time_step,
                "avg_global_time_step": avg_global_time_step,
                "total_planning_and_exec_time": total_planning_and_exec_time,
                "tcp_max_speed": tcp_max_speed,
                "distance_to_goal": distance_to_goal,
                "distance_to_nearest_obstacle": distance_to_nearest_obstacle,
                "total_joints_effort": total_joints_effort,
                "n_timestep": n_timestep,
                "smoothness": smoothness,
                "total_execution_time": total_execution_time,
                "lp_infeasible": lp_infeasible,
                "manipulability": manipulability
            }
            df = df.append(pd.Series(episodes_stats), ignore_index=True)

            if df_path is not None:
                df.to_csv(df_path, index=False)

        df = df.assign(algorithm=exp_name, scenario=scenario_)
    return df


def scenario_to_df_for_dynamic(episodes, exp_dir=None, scenario_=None, load_pd_if_Exists=False, exp_name="", **kwargs):
    if exp_dir is not None and scenario_ is not None:
        df_path = join(exp_dir, "pd_" + scenario_ + ".csv")
    else:
        df_path = None

    if load_pd_if_Exists and df_path is not None and os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        episodes_stats_keys = ["distance_to_nearest_obstacle", "distance_to_goal", "tcp_speed", "goal_speed",
                               "relative_robot_to_goal_speed"]
        #         to_plot = [self.distance_to_nearest_obstacle, self.distance_to_goal, self.tcp_speed, self.goal_speed,
        #                    self.relative_robot_to_goal_speed]
        df = pd.DataFrame(columns=episodes_stats_keys)
        for iepisode, e in enumerate(episodes):
            # e.plot_properties(save_dir=output_dir, n_episode=iepisode)
            # e.plot_properties(n_episode=iepisode)
            distance_to_nearest_obstacle = e.distance_to_nearest_obstacle
            distance_to_goal = e.distance_to_goal
            tcp_speed = e.tcp_speed
            goal_speed = e.goal_speed
            relative_robot_to_goal_speed = e.relative_robot_to_goal_speed
            min_length = min(len(distance_to_goal), len(distance_to_nearest_obstacle), len(tcp_speed), len(goal_speed),
                             len(relative_robot_to_goal_speed), )
            episodes_stats = pd.DataFrame({
                "distance_to_goal": distance_to_goal[:min_length],
                "distance_to_nearest_obstacle": distance_to_nearest_obstacle[:min_length],
                "tcp_speed": tcp_speed[:min_length],
                "goal_speed": goal_speed[:min_length],
                "relative_robot_to_goal_speed": relative_robot_to_goal_speed[:min_length],
            })
            # df = df.append(pd.Series(episodes_stats), ignore_index=True)

            df = pd.concat([df, episodes_stats])

        df = df.assign(algorithm=exp_name, scenario=scenario_)
    return df


def exp_to_df(exp_dir, exp_name="", is_dynamic=False, **kwargs):
    df_exp = None
    for iscenario, scenario_ in enumerate(tqdm(scenarios, desc=exp_name, leave=True)):
        files = glob.glob(join(exp_dir, f"*_{scenario_}_*.pb"))
        if len(files) == 0:
            continue
        episodes = []
        for f in files:
            e = load_object(path=f)
            if e is None:
                continue
            else:
                episodes.extend(e)
        if not is_dynamic:
            df = scenario_to_df(episodes, scenario_=scenario_, exp_name=exp_name, exp_dir=exp_dir, **kwargs)
        else:
            df = scenario_to_df_for_dynamic(episodes, scenario_=scenario_, exp_name=exp_name, exp_dir=exp_dir, **kwargs)
        if df_exp is None:
            df_exp = df
        else:
            df_exp = pd.concat([df_exp, df])
    if df_exp is None:
        # check if already a pandas csv file exists (results were executed on a different maching
        existing_pd_path = join(exp_dir, "gen_exp_dataframes.csv")
        if os.path.exists(existing_pd_path):
            df_exp = pd.read_csv(existing_pd_path)
    return df_exp  # single exp with all scenarios


def all_to_df(exp_dir, **kwargs):
    exp_dirs_all = glob.glob(join(exp_dir, "*/"))
    df_all = None
    for iexp, exp_dir_ in enumerate(exp_dirs_all):
        name = os.path.basename(os.path.dirname(exp_dir_))
        if len(name) < 13:
            pass
        else:
            name = os.path.basename(exp_dir_).split("_")
            name = "".join(n[0] for n in name[:3])

        df_exp = exp_to_df(exp_dir_, exp_name=name, **kwargs)
        if df_all is None:
            df_all = df_exp
        else:
            df_all = pd.concat([df_all, df_exp])

    return df_all


if __name__ == '__main__':
    # %%
    # exp_dir = "/home/mohamed/git/free_space/robot_control/planner/exp_generator/experiments/Results/static_environment"
    # exp_dir = "/home/mohamed/git/free_space/experiments/glir/Results/analysis"
    # exp_dir = "/home/mohamed/git/free_space/robot_control/planner/video/analysis"
    # exp_dir = "/home/mohamed/git/free_space/experiments/glir/Results/analysis_2"
    # exp_dir = "/home/mohamed/git/free_space/experiments/glir/Results_corrected_distance/glir_vrep_0.25"
    # exp_dir = "/home/mohamed/git/free_space/experiments/glir/Results_corrected_distance/analysis"
    # exp_dir = "/home/mohamed/git/glir/experiments/Results_analysis/corrected_octomap_5cm/moveit"
    # exp_dir = "/home/mohamed/git/glir/experiments/Results_analysis/corrected_octomap_2.5cm/moveit"

    # exp_dir = "/home/mohamed/git/glir/experiments/Results_analysis/corrected_octomap_0.025_col/moveit"
    # exp_dir = r"C:\Users\mae\git\glir\experiments\00_Results\glirv3"
    # exp_dir = r"C:\Users\mae\git\demonstrator\playground\check_simulation\Results_test\glir_mesh_grid_only_local"
    # exp_dir = r"../static_exp/moveit/Results"
    is_dynamic = False
    exp_dir = r"../static_exp/glir/glir"
    exp_dir, is_dynamic = r"../static_obstacle_dynamic_goal_exp/dynamic_goal_hard", True

    # exp_dir = "/home/mohamed/git/free_space/experiments/glir/Results_corrected_distance/moveit"
    # exp_dir = "/home/mohamed/git/free_space/robot_control/planner/exp_generator/experiments/Results/test"
    # df = all_to_df(exp_dir, dt=None, goal_tol=1e-3)
    df = all_to_df(exp_dir, dt=None, goal_tol=3e-2, is_dynamic=is_dynamic)
    df.head()
    # max_joint_speed
    # max_joint_jerk
    # infeasible: planner failed? -> emergency stop triggered
    if is_dynamic:
        df.to_csv(join(exp_dir, "gen_exp_dataframes.csv"))
    else:
        sub_df = df[[
            "algorithm",
            "scenario",
            "is_safe",
            "is_not_safe",
            "is_safe_and_success",
            # "is_unsafe_and_success",
            "goal_success",
            # "is_touching_speed",
            # "is_hitting_speed",
            # "is_hitting_distance",
            # "is_touching_distance",
            "distance_to_nearest_obstacle",
            "distance_to_goal",
            "avg_local_time_step",
            "avg_global_time_step",
            "total_planning_and_exec_time",
            "total_execution_time",
            "geometric_time",
            "total_global_query_time",
            # "total_local_time_step",
            # "total_global_time",
            "total_joints_effort",
            "n_timestep",
            "smoothness",
            "lp_infeasible",
            "tcp_max_speed",
            "manipulability",
        ]]

        sub_df.to_csv(join(exp_dir, "gen_exp_dataframes.csv"))
