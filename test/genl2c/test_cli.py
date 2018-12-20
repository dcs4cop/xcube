import unittest
from typing import List


import click
import click.testing

from xcube.genl2c.cli import cli


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

    def test_main_with_illegal_size_option(self):
        result = self.invoke_cli(['-s', '120,abc', 'input.nc'])
        self.assertEqual(2, result.exit_code)

    def test_main_with_illegal_region_option(self):
        result = self.invoke_cli(['-r', '50,_2,55,21', 'input.nc'])
        self.assertEqual(2, result.exit_code)
        result = self.invoke_cli(['-r', '50,20,55', 'input.nc'])
        self.assertEqual(2, result.exit_code)

    def test_main_with_illegal_variable_option(self):
        result = self.invoke_cli(['-v', ' ', 'input.nc'])
        self.assertEqual(2, result.exit_code)

    def test_main_with_illegal_config_option(self):
        result = self.invoke_cli(['-c', 'nonono.yml', 'input.nc'])
        self.assertEqual(2, result.exit_code)

    def test_main_with_illegal_options(self):
        result = self.invoke_cli(['input.nc'])
        self.assertEqual(2, result.exit_code)

    def test_version(self):
        result = self.invoke_cli(['--version'])
        print(result.output)
        print(result.exit_code)
        self.assertEqual(
        f'{prog!r}, version 0.1.0.dev1', result.output)