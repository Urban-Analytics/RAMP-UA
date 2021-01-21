import os
import unittest
from unittest.mock import patch, mock_open
from microsim.utilities import download_data

class TestDownloadData(unittest.TestCase):

    def test_download_success(self):
        """
        A test to fetch a dummy data tar file using the download data function in utils
        """
        open_mock = mock_open()

        with patch("builtins.open", open_mock):
            download_data(url="https://example0blob0store.blob.core.windows.net/test1/dummy_data.tar.gz")
        
        open_mock.assert_called_with("devon_data.tar.gz","wb")

    def test_download_fail(self):
        """
        A test the confirms an exception is raised if an invalid url is provided to download data
        """

        with self.assertRaises(Exception):
            download_data(url="not_a_url")

if __name__ == '__main__':
    unittest.main()