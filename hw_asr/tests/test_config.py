import json
import unittest

# import sys
# sys.path.append("/home/yubuntu/asr_project_template")

from hw_asr.tests.utils import clear_log_folder_after_use
from hw_asr.utils.parse_config import ConfigParser


class TestConfig(unittest.TestCase):
    def test_create(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            json.dumps(config_parser.config, indent=2)
