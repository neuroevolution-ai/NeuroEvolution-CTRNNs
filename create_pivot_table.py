import numpy as np
import pandas as pd


def apply_global_filters(data: pd.DataFrame) -> pd.DataFrame:
    filtered_data = data[data["experiment_id"] == 1]
    filtered_data = filtered_data[filtered_data["neural_network_type"] == "CTRNN"]

    return filtered_data


def create_pivot_table_row(data: pd.DataFrame, row_property: str, environments: list) -> pd.DataFrame:
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
            columns=["maximum_average_mean", "maximum_average_amax", "maximum_average_len", "elapsed_time_mean"])

        per_env_pivot_tables.append(env_pivot_table)

    # Concatenate the individual pivot tables horizontally, i.e. this will be one row in the final pivot table
    pivot_table_row = pd.concat(per_env_pivot_tables, axis=1)

    return pivot_table_row


def main():
    experiments_data = pd.read_csv("spreadsheets/output.csv")

    # Filters used for each part of the pivot table (e.g. neural network used is CTRNN, ...)
    globally_filtered_data = apply_global_filters(experiments_data)
    environments = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2"]
    row_properties = ["population_size", "number_neurons", "clipping_range", "optimize_y0",
                      "set_principle_diagonal_elements_of_W_negative", "sigma"]

    rows = []

    for row_prop in row_properties:
        rows.append(create_pivot_table_row(globally_filtered_data,
                                           row_property=row_prop,
                                           environments=environments))

    # Create the overall pivot table
    pivot_table = pd.concat(rows, keys=row_properties, axis=0)

    pivot_table.to_latex("spreadsheets/pivot_table.tex")


if __name__ == "__main__":
    main()
