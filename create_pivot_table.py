import numpy as np
import pandas as pd


def apply_global_filters(data: pd.DataFrame) -> pd.DataFrame:
    filtered_data = data[data["experiment_id"] == 1]
    filtered_data = filtered_data[filtered_data["neural_network_type"] == "CTRNN"]

    return filtered_data


def round_pivot_table(pivot_table: pd.DataFrame, environments: list):
    # Integers look better in the paper therefore we first round and then cast to remove trailing zeros (except for
    # the average time)
    for env in environments:
        pivot_table[env] = pivot_table[env].round({"maximum_average_mean": 0,
                                                   "maximum_average_amax": 0,
                                                   "maximum_average_len": 0,
                                                   "elapsed_time_mean": 1})

        # TODO this will for some reason not work
        # pivot_table[env] = pivot_table[env].astype({"maximum_average_mean": np.int32,
        #                                             "maximum_average_amax": np.int32,
        #                                             "maximum_average_len": np.int32,
        #                                             "elapsed_time_mean": np.float32})

    return pivot_table


def create_pivot_table_row(data: pd.DataFrame, row_property: str, environments: list,
                           order_of_columns: list) -> pd.DataFrame:
    per_env_pivot_tables = []

    for env in environments:
        locally_filtered_data = data[data["environment"] == env]
        env_pivot_table = pd.pivot_table(locally_filtered_data,
                                         values=["maximum_average", "elapsed_time"],
                                         index=[row_property],
                                         aggfunc={"maximum_average": [np.mean, np.max, len], "elapsed_time": np.mean})

        # This simply flattens the column names because the pivot table functions returns them in a hierarchy
        env_pivot_table.columns = env_pivot_table.columns.to_series().str.join("_")

        # Now explicitly reorder the columns to be sure that they are in correct order
        env_pivot_table = env_pivot_table.reindex(
            columns=order_of_columns)

        env_pivot_table["elapsed_time_mean"] = env_pivot_table["elapsed_time_mean"] / 3600

        per_env_pivot_tables.append(env_pivot_table)

    # Concatenate the individual pivot tables horizontally, i.e. this will be one row in the final pivot table
    pivot_table_row = pd.concat(per_env_pivot_tables, keys=environments, axis=1)

    return pivot_table_row


def add_total_row(pivot_table: pd.DataFrame, environments: list, order_of_columns: list,
                  total_row_functions: list) -> pd.DataFrame:
    # This simply says apply this function to that column, i.e. apply max to the column where we list max values
    # because we want the maximum reward over all the rows in the pivot table
    aggregate_functions = {col: col_fun for (col, col_fun) in zip(order_of_columns, total_row_functions)}

    total_row = []

    for env in environments:
        per_env_data = pivot_table[env]

        total_row.append(per_env_data.agg(aggregate_functions))

    total_row_values = pd.concat(total_row).values

    pivot_table.loc["Total", :] = total_row_values

    return pivot_table


def main():
    experiments_data = pd.read_csv("spreadsheets/output.csv")

    # Filters used for each part of the pivot table (e.g. neural network used is CTRNN, ...)
    globally_filtered_data = apply_global_filters(experiments_data)
    environments = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2"]
    row_properties = ["population_size", "number_neurons", "clipping_range", "optimize_y0",
                      "set_principle_diagonal_elements_of_W_negative", "sigma"]
    column_order = ["maximum_average_mean", "maximum_average_amax", "maximum_average_len", "elapsed_time_mean"]
    # This has to match column_order, as these functions will be used on the columns to calculate the total row values
    total_row_functions = [np.mean, np.max, len, np.mean]

    rows = []

    for row_prop in row_properties:
        rows.append(create_pivot_table_row(globally_filtered_data,
                                           row_property=row_prop,
                                           environments=environments,
                                           order_of_columns=column_order))

    # Create the overall pivot table
    pivot_table = pd.concat(rows, keys=row_properties, axis=0)

    pivot_table = add_total_row(pivot_table, environments, column_order, total_row_functions)

    pivot_table = round_pivot_table(pivot_table, environments)

    pivot_table.to_latex("spreadsheets/pivot_table.tex")


if __name__ == "__main__":
    main()
