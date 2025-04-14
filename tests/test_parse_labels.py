import unittest
import os
from label import parse_labels, Label

class TestParseLabels(unittest.TestCase):

    def test_parse_labels_yaml(self):
        labels_dict = parse_labels('./tests/labelmap.yml')
        self.assertEqual(len(labels_dict), 3)
        self.assertEqual(labels_dict[1], 'Label One')
        self.assertEqual(labels_dict[2], 'Label Two')
        self.assertEqual(labels_dict[3], 'Label Three')

    def test_parse_labels_text(self):
        labels_dict = parse_labels('./tests/labelmap.txt')
        self.assertEqual(len(labels_dict), 3)
        self.assertEqual(labels_dict[1], 'Label One')
        self.assertEqual(labels_dict[2], 'Label Two')
        self.assertEqual(labels_dict[3], 'Label Three')

if __name__ == '__main__':
    unittest.main()