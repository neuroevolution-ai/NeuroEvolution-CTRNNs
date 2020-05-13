import unittest
import numpy as np

import argparse
from unittest import mock  # python 3.3+
import importlib


class MyTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass



    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(configuration="test_configuration_cma.json", from_checkpoint=None))
    def test_CMA(self,mocks):
        import CTRNN_train
        importlib.reload(CTRNN_train)
        configuration_data = CTRNN_train.configuration_data
        stats = CTRNN_train.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        pop, log = CTRNN_train.trainer.train(stats,
                                             number_generations=configuration_data["number_generations"],
                                             checkpoint=None,
                                             cb_before_each_generation=CTRNN_train.reset_seed_for_generation, )
        self.assertEqual(log[-1]["max"], -101.51086159476436)


    @mock.patch('argparse.ArgumentParser.parse_args',
                return_value=argparse.Namespace(configuration="test_configuration_mu_lambda.json", from_checkpoint=None))
    def test_mu_lambda(self,mocks):
        import CTRNN_train
        importlib.reload(CTRNN_train)
        configuration_data = CTRNN_train.configuration_data
        stats = CTRNN_train.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        pop, log = CTRNN_train.trainer.train(stats,
                                             number_generations=configuration_data["number_generations"],
                                             checkpoint=None,
                                             cb_before_each_generation=CTRNN_train.reset_seed_for_generation, )
        self.assertEqual(log[-1]["max"], -104.93246283147762)

if __name__ == '__main__':
    unittest.main()
