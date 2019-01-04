import unittest
from typing import List

import click
import click.testing

from xcube.genl3.cli import cli


class CliTest(unittest.TestCase):
    @classmethod
    def invoke_cli(cls, args: List[str]):
        runner = click.testing.CliRunner()
        return runner.invoke(cli, args)

    def test_main_with_option_help(self):
        result = self.invoke_cli(['--help'])
        self.assertEqual(0, result.exit_code)

    def test_main_with_missing_args(self):
        result = self.invoke_cli([])
        # exit_code==0 when run from PyCharm, exit_code==2 else
        self.assertEqual(2, result.exit_code)
        # self.assertTrue(exit_code in [0, 2])

    def test_variable_option(self):
        result = self.invoke_cli(['-v', ' ', 'input.nc'])
        self.assertEqual(2, result.exit_code)

    def test_metadata_option(self):
        result = self.invoke_cli(['-m', 'nonono.yml', 'input.nc'])
        self.assertEqual(2, result.exit_code)

    def test_resampling_option(self):
        result = self.invoke_cli(['--resampling', 'NeArEsT'])
        self.assertEqual(2, result.exit_code)

    def test_illegal_input(self):
        result = self.invoke_cli(['input.nc'])
        self.assertEqual(2, result.exit_code)
