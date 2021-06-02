import os
import json
import argparse
import logging
import csv
import pandas as pd
from typing import List, Tuple, Union
import numpy as np


def read_simulations(base_directory):
    simulation_runs = []
    for d in os.listdir(base_directory):
        simulation_folder = os.path.join(base_directory, d)

        # noinspection PyBroadException
        try:
            with open(os.path.join(simulation_folder, 'Configuration.json'), "r") as read_file:
                conf = json.load(read_file)
        except Exception:
            conf = None
            logging.error("couldn't read conf for " + str(simulation_folder), exc_info=True)
        # noinspection PyBroadException
        try:
            with open(os.path.join(simulation_folder, 'Log.json'), 'r') as read_file:
                log = json.load(read_file)
        except Exception:
            log = None
            logging.error("couldn't read log for " + str(simulation_folder), exc_info=True)
        sim = {
            "dir": d,
            "conf": conf,
            "log": log,
        }
        simulation_runs.append(sim)
    return simulation_runs


def get_attribute_or_none(d, attr):
    if attr in d:
        return d[attr]
    return None


def gather_info_for_csv(simulation, conf_keys):
    log = simulation["log"]
    conf = simulation["conf"]
    avg = [generation["avg"] for generation in log]
    maximum = [generation["max"] for generation in log]

    info = dict()

    for key in conf_keys:
        info[key] = get_attribute_or_none(conf, key)

        if key == "clipping_range" and info[key] is None:
            info[key] = False

    info["maximum_average"] = max(avg)
    info["maximum"] = max(maximum)
    info["directory"] = simulation["dir"]

    return info


def generate_filter(experiment_data: pd.DataFrame, filters: List[Tuple], bitwise_and=True, existing_filters=None):
    """
    Combine a list of column names either wit bitwise and or bitwise or to filter the data of experiment_data

    Bitwise and combination means that all the filters defined are chained together and the data that is left must
    correspond to all the filters. With bitwise or it is enough that only one of the filters evaluates to 'True'

    :param experiment_data: DataFrame on the data which shall be filtered
    :param filters: List of tuples, where the first entry is the column name and the second the value that this column
                    shall have
    :param bitwise_and: When True, then the filters are combined with a bitwise and, when False with bitwise or.
    :param existing_filters: Useful if you already have generated filters and now want to additionally add some
    :return: A list with the generated filters. This list can be used to access the experiment_data to get the filtered
             values. This means that this function only returns the filters and not filtered data.
    """
    last_filter = None

    if existing_filters is not None:
        last_filter = existing_filters

    for (label, value) in filters:
        current_filter = experiment_data[label] == value

        if last_filter is None:
            last_filter = current_filter
            continue

        if bitwise_and:
            last_filter = last_filter & current_filter
        else:
            # Use bitwise or, used if multiple values per column shall be used
            last_filter = last_filter | current_filter

    return last_filter


def combine_columns(experiment_data: pd.DataFrame, columns: List[str], unpack_last_column=False):
    altered_data = experiment_data.copy()

    if unpack_last_column:
        altered_data[columns[-1]] = experiment_data[columns[-1]].apply(lambda x: x if np.isnan(x) else np.sum(x))

    assert len(columns) >= 2

    for col in columns[1:]:
        altered_data[columns[0]] = altered_data[columns[0]].fillna(altered_data[col])

    altered_data = altered_data.drop(columns=columns[1:])

    return altered_data


def create_pivot_table(table: pd.DataFrame, group_by_column: Union[List[str], str], columns: List,
                       filters=None):
    """
    Creates a pivot table by first filtering the data with an optional provided filter and then grouping the data
    according to 'group_by_column'. Then the mean and max is taken on the grouped data. To ensure that both mean and
    max columns are distinguishable the columns get renamed to colname_mean and colname_max. Then the count of values,
    the mean and the max data is concatenated and returned.

    group_by_column can also be a List of two columns but this assumes that these two columns are describing essentially
    the same data. For example clipping_range_min and clipping_range_max is such a case. Then it is checked if both
    columns have the same amount of data and then only clipping_range_min will be taken to do the grouping.

    :param table: The data on which the pivot table is created
    :param group_by_column: The column that is grouped by
    :param columns: A list of columns that shall be present in the pivot table, this must contain the group_by_column
    :param filters: Optional filters that can be applied to the data
    :return:
    """
    filtered_data = table

    if filters is not None:
        filtered_data = table[filters]

    filtered_data = filtered_data[columns]

    if isinstance(group_by_column, list):
        # This case is for clipping_min and clipping_max which will be displayed as one part of the table
        # "Clipping [-x, x]"
        assert len(group_by_column) == 2
        assert (filtered_data.groupby(group_by_column[0]).size().values ==
                filtered_data.groupby(group_by_column[1]).size().values)

        # Remove one of the group vales to not mess up with the other pivot tables
        filtered_data = filtered_data.drop(group_by_column[1], axis=1)

        grouped_data = filtered_data.groupby(group_by_column[0])
    else:
        grouped_data = filtered_data.groupby(group_by_column)

    output_mean = grouped_data.mean().rename(columns={_col: _col + "_mean" for _col in columns})
    output_max = grouped_data.max().rename(columns={_col: _col + "_max" for _col in columns})
    output_count = pd.DataFrame(grouped_data.size()).rename(columns={0: "Count"})

    # First column: Count, second mean over best rewards, third max over best rewards
    output = pd.concat([output_mean.iloc[:, 0], output_max.iloc[:, 0], output_count, output_mean.iloc[:, 1]], axis=1)

    # file_name = "".join([str(value) + "-" for _, value in filters]) + "---" + str(group_by_column) + ".csv"
    # output_mean.to_csv(os.path.join(dir_path, file_name))

    return output


def create_merged_pivot_tables(experiment_data, environment: str):
    filters = [("neural_network_type", "CTRNN"), ("environment", environment)]
    filters = generate_filter(experiment_data, filters, bitwise_and=True)

    base_columns = ["maximum_average"]

    pivot_table_population_size = create_pivot_table(experiment_data,
                                                     group_by_column="population_size",
                                                     columns=base_columns + ["population_size", "elapsed_time"],
                                                     filters=filters)

    pivot_table_number_neurons = create_pivot_table(experiment_data,
                                                    group_by_column="number_neurons",
                                                    columns=base_columns + ["number_neurons", "elapsed_time"],
                                                    filters=filters)

    pivot_table_clipping_range = create_pivot_table(experiment_data,
                                                    group_by_column="clipping_range",
                                                    columns=base_columns + ["clipping_range", "elapsed_time"],
                                                    filters=filters)

    pivot_table_optimize_y0 = create_pivot_table(experiment_data,
                                                 group_by_column="optimize_y0",
                                                 columns=base_columns + ["optimize_y0", "elapsed_time"],
                                                 filters=filters)

    pivot_table_diagonal_negative = create_pivot_table(experiment_data,
                                                       group_by_column="set_principle_diagonal_elements_of_W_negative",
                                                       columns=base_columns + [
                                                           "set_principle_diagonal_elements_of_W_negative",
                                                           "elapsed_time"],
                                                       filters=filters)

    pivot_table_sigma = create_pivot_table(experiment_data,
                                           group_by_column="sigma",
                                           columns=base_columns + ["sigma", "elapsed_time"],
                                           filters=filters)

    merged_pivots = pd.concat([pivot_table_population_size,
                               pivot_table_number_neurons,
                               pivot_table_clipping_range,
                               pivot_table_optimize_y0,
                               pivot_table_diagonal_negative,
                               pivot_table_sigma],
                              keys=["Population size", "Number neurons", "Clipping range", "optimize_x0",
                                    "W_xx_negative", "Initial Deviation"])

    return merged_pivots


def create_total_rows(from_dataframe):
    total_row = from_dataframe.groupby(level=0).agg(
        maximum_average_mean=("maximum_average_mean", "mean"),
        maximum_average_max=("maximum_average_max", "max"),
        Count=("Count", "sum"),
        elapsed_time_mean=("elapsed_time_mean", "mean")).reset_index().mean()

    return total_row


def generate_ctrnn_pivot_table(_dir_path, experiment_data: pd.DataFrame):

    pivots_hopper = create_merged_pivot_tables(experiment_data, "Hopper-v2")
    pivots_cheetah = create_merged_pivot_tables(experiment_data, "HalfCheetah-v2")
    pivots_walker = create_merged_pivot_tables(experiment_data, "Walker2d-v2")

    total_row_hopper = create_total_rows(pivots_hopper)
    total_row_cheetah = create_total_rows(pivots_cheetah)
    total_row_walker = create_total_rows(pivots_walker)

    concatenated_tables = pd.concat({"Hopper-v2": pivots_hopper,
                                     "HalfCheetah-v2": pivots_cheetah,
                                     "Walker2d-v2": pivots_walker}, axis=1)

    concatenated_total_rows = pd.concat([total_row_hopper.to_frame().T,
                                         total_row_cheetah.to_frame().T,
                                         total_row_walker.to_frame().T], axis=1)

    # Use round + casting to integer because only casting does no proper rounding and without casting the values
    # have .0 as a decimal place
    concatenated_tables = concatenated_tables.round().astype(int)

    cleaner_column_names = {"maximum_average_mean": "Reward Mean",
                            "maximum_average_max": "Reward Max",
                            "elapsed_time_mean": "Avg Time"}

    concatenated_tables = concatenated_tables.rename(columns=cleaner_column_names)

    concatenated_total_rows.to_csv(os.path.join(_dir_path, "ctrnn_pivot_table.csv"))

    # Pandas has a nice feature to generate LaTeX tables from a DataFrame
    concatenated_tables.to_latex(os.path.join(_dir_path, "ctrnn_pivot_table.tex"))

    concatenated_total_rows = concatenated_total_rows.round().astype(int)
    concatenated_total_rows = concatenated_total_rows.rename(columns=cleaner_column_names)

    concatenated_total_rows.to_csv(os.path.join(_dir_path, "ctrnn_total_row.csv"))
    concatenated_total_rows.to_latex(os.path.join(_dir_path, "ctrnn_total_row.tex"))


logging.basicConfig()
parser = argparse.ArgumentParser(description='Visualize experiment results')
parser.add_argument('--dir', metavar='dir', type=str, help='base directory for input',
                    default='Simulation_Results')
parser.add_argument('--csv', metavar='type', type=str, help='location of output csv file', default='output.csv')
args = parser.parse_args()

configuration_keys = []
for simulation in read_simulations(args.dir):
    for key in simulation["conf"].keys():
        if key not in configuration_keys:
            configuration_keys.append(key)

# configuration_keys.remove("random_seed_for_environment")
# configuration_keys.remove("use_biases")
# configuration_keys.remove("delta_t")

header = configuration_keys
header.extend(["maximum_average", "maximum", "directory"])

data = []
for simulation in read_simulations(args.dir):
    data.append(gather_info_for_csv(simulation, configuration_keys))

experiments_dataframe = pd.DataFrame(data)
experiments_dataframe = experiments_dataframe[experiments_dataframe["experiment_id"] == 1]

dir_path = "spreadsheets"
os.makedirs(dir_path, exist_ok=True)

generate_ctrnn_pivot_table(_dir_path=dir_path, experiment_data=experiments_dataframe)

with open(args.csv, 'w') as output_file:
    dict_writer = csv.DictWriter(output_file, header)
    dict_writer.writeheader()
    dict_writer.writerows(data)
