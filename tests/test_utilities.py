import os
import unittest
from unittest.mock import patch, mock_open
from microsim.utilities import download_data, unpack_data, data_setup

class TestDownloadData(unittest.TestCase):

    @patch("microsim.utilities.open")
    def test_download_success(self, open_mock):
        """
        A test to fetch a dummy data tar file using the download data function in utils
        """

        tarName = download_data(url="https://ramp0storage.blob.core.windows.net/rampdata/dummy_data.tar.gz")
        
        open_mock.assert_called_with("dummy_data.tar.gz","wb")
        self.assertTrue(tarName == 'dummy_data.tar.gz')

    def test_download_fail(self):
        """
        A test the confirms an exception is raised if an invalid url is provided to download data
        """

        with self.assertRaises(Exception):
            download_data(url="not_a_url")

    @patch("microsim.utilities.tarfile")
    def test_unpack_data(self, mock_tar):
        """
        A test of the unpack_data function using mocks to check tarfile functions are called
        """

        unpack_data("example_tar")

        mock_tar.open.assert_called_with("example_tar")
        mock_tar.open().extractall.assert_called_with(".")

    @patch("microsim.utilities.download_data")
    @patch("microsim.utilities.unpack_data")
    def test_data_setup(self, mock_ud, mock_dd):

        data_setup(archive = 'devon_data')

        mock_dd.assert_called_with(url="https://ramp0storage.blob.core.windows.net/rampdata/devon_data.tar.gz")
        mock_ud.assert_called_with(archive = "devon_data")


if __name__ == '__main__':
    unittest.main()