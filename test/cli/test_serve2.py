from test.cli.helpers import CliTest


class ServerCliTest(CliTest):
    def test_help(self):
        result = self.invoke_cli(["serve2", "--help"])
        self.assertEqual(0, result.exit_code)

    def test_update_after(self):
        result = self.invoke_cli(["serve2",
                                  "--update-after", "0.1",
                                  "--stop-after", "0.2"])
        self.assertEqual(0, result.exit_code)