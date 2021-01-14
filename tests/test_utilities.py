import os
import unittest
from unittest.mock import patch, mock_open
from microsim.utilities import download_data

class TestDownloadData(unittest.TestCase):

    def test_download_success(self):
        open_mock = mock_open()

        with patch("__main__.open", open_mock):
            download_data()
        
        open_mock.assert_called_with("devon_data.tar.gz","wb")

    def test_download_fail(self):

        with self.assertRaises(Exception):
            download_data(url="not_a_url")

if __name__ == '__main__':
    unittest.main()