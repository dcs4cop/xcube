import json
import unittest

from xcube.core.gen2.service.config import ServiceConfig


class ServiceConfigTest(unittest.TestCase):

    def test_from_dict(self):
        json_instance = dict(endpoint_url='https://stage.xcube-gen.brockmann-consult.de/api/v2',
                             access_token='02945ugjhklojg908ijr023jgbpij202jbv00897v0798v65472')
        service_config = ServiceConfig.from_dict(json_instance)
        self.assertIsInstance(service_config, ServiceConfig)
        self.assertEqual('https://stage.xcube-gen.brockmann-consult.de/api/v2/', service_config.endpoint_url)
        self.assertEqual('02945ugjhklojg908ijr023jgbpij202jbv00897v0798v65472', service_config.access_token)
        self.assertEqual(None, service_config.client_id)
        self.assertEqual(None, service_config.client_secret)

    def test_to_dict(self):
        expected_dict = dict(endpoint_url='https://stage.xcube-gen.brockmann-consult.de/api/v2/',
                             access_token='02945ugjhklojg908ijr023jgbpij202jbv00897v0798v65472')
        service_config = ServiceConfig.from_dict(expected_dict)
        actual_dict = service_config.to_dict()
        self.assertEqual(expected_dict, actual_dict)
        # smoke test JSON serialisation
        json.dumps(actual_dict, indent=2)
