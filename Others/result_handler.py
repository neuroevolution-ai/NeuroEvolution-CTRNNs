
import os
import json
import pickle


class ResultHandler(object):

    def __init__(self, startDate, nn_type, configuration_data, base_path="Simulation_Results"):
        self.base_path = base_path
        self.startDate = startDate
        self.nn_type = nn_type
        self.configuration_data = configuration_data

    def write_result(self, hof, log, time_elapsed, individual_size, input_size, output_size):
        # Create new directory to store data of current simulation run
        directory = os.path.join(self.base_path, self.nn_type, self.startDate)
        os.makedirs(directory)
        print("output directory: " + str(directory))
        configuration_data = self.configuration_data
        # Save Configuration file as json
        with open(os.path.join(directory, 'Configuration.json'), 'w') as outfile:
            json.dump(self.configuration_data, outfile)

        # Save hall of fame individuals
        with open(os.path.join(directory, 'HallOfFame.pickle'), "wb") as fp:
            pickle.dump(hof, fp)

        # Save Log
        with open(os.path.join(directory, 'Log.json'), 'w') as outfile:
            json.dump(log, outfile)

        # Write Log to text file
        with open(os.path.join(directory, 'Log.txt'), 'w') as write_file:
            write_file.write('Gym environment: {:s}\n'.format(configuration_data["environment"]))
            write_file.write(
                'Random seed for environment: {:d}\n'.format(configuration_data["random_seed_for_environment"]))
            write_file.write('Number of neurons: {:d}\n'.format(configuration_data["number_neurons"]))
            write_file.write('Time step (delta_t): {:.3f}\n'.format(configuration_data["delta_t"]))
            write_file.write('Optimize state boundaries: {}\n'.format(configuration_data["optimize_state_boundaries"]))
            write_file.write('Clipping range max: {:.2f}\n'.format(configuration_data["clipping_range_max"]))
            write_file.write('Clipping range min: {:.2f}\n'.format(configuration_data["clipping_range_min"]))
            write_file.write('Optimize initial states y0: {}\n'.format(configuration_data["optimize_y0"]))
            write_file.write('Set principal elements of W to negative: {}\n'.format(
                configuration_data["set_principle_diagonal_elements_of_W_negative"]))
            write_file.write('Population size: {:d}\n'.format(configuration_data["population_size"]))
            write_file.write('Number of Generations: {:d}\n'.format(configuration_data["number_generations"]))
            write_file.write('Sigma: {:.2f}\n'.format(configuration_data["sigma"]))
            write_file.write('Number of runs per evaluation: {:d}\n'.format(configuration_data["number_fitness_runs"]))
            write_file.write('\n')
            write_file.write('Genome Size: {:d}\n'.format(individual_size))
            write_file.write('Inputs: {:d}\n'.format(input_size))
            write_file.write('Outputs: {:d}\n'.format(output_size))
            write_file.write('\n')
            write_file.write(
                'Number of neurons in hidden layer 1: {:d}\n'.format(configuration_data["number_neurons_layer1"]))
            write_file.write(
                'Number of neurons in hidden layer 2: {:d}\n'.format(configuration_data["number_neurons_layer2"]))
            write_file.write('\n')

            dash = '-' * 80

            write_file.write(dash + '\n')
            write_file.write(
                '{:<8s}{:<12s}{:<16s}{:<16s}{:<16s}{:<16s}\n'.format('gen', 'nevals', 'avg', 'std', 'min', 'max'))
            write_file.write(dash + '\n')

            # Write data for each episode
            for line in log:
                write_file.write(
                    '{:<8d}{:<12d}{:<16.2f}{:<16.2f}{:<16.2f}{:<16.2f}\n'.format(line['gen'], line['nevals'],
                                                                                 line['avg'], line['std'], line['min'],
                                                                                 line['max']))

            # Write elapsed time
            write_file.write("\nTime elapsed: %.4f seconds" % (time_elapsed))
