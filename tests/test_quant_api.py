import os
import unittest
import pandas as pd
import numpy as np
from microsim.quant_api import QuantRampAPI

test_dir = os.path.dirname(os.path.abspath(__file__))

class TestDownloadData(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.working_quant_class = QuantRampAPI(quant_dir=os.path.join(test_dir, 'dummy_data', 'QUANT-test'))

    def test_class_init(self):
        """A test to confirm that the class instantiation has set some object attributes
        """

        expected_attrs = ['dfPrimaryPopulation',
                            'dfPrimaryZones',
                            'primary_probPij',
                            'dfSecondaryPopulation',
                            'dfSecondaryZones',
                            'secondary_probPij',
                            'dfRetailPointsPopulation',
                            'dfRetailPointsZones',
                            'retailpoints_probSij',
                            'dfHospitalPopulation',
                            'dfHospitalZones',
                            'hospital_probHij']

        self.assertTrue(set(expected_attrs).issubset(self.working_quant_class.__dir__()))

        # confirm expected types
        self.assertTrue(isinstance(self.working_quant_class.dfPrimaryPopulation, pd.DataFrame))
        self.assertTrue(isinstance(self.working_quant_class.dfPrimaryZones, pd.DataFrame))
        self.assertTrue(isinstance(self.working_quant_class.primary_probPij, np.ndarray))
        self.assertTrue(isinstance(self.working_quant_class.dfSecondaryPopulation, pd.DataFrame))
        self.assertTrue(isinstance(self.working_quant_class.dfSecondaryZones, pd.DataFrame))
        self.assertTrue(isinstance(self.working_quant_class.secondary_probPij, np.ndarray))
        self.assertTrue(isinstance(self.working_quant_class.dfRetailPointsPopulation, pd.DataFrame))
        self.assertTrue(isinstance(self.working_quant_class.dfRetailPointsZones, pd.DataFrame))
        self.assertTrue(isinstance(self.working_quant_class.retailpoints_probSij, np.ndarray))
        self.assertTrue(isinstance(self.working_quant_class.dfHospitalPopulation, pd.DataFrame))
        self.assertTrue(isinstance(self.working_quant_class.dfHospitalZones, pd.DataFrame))
        self.assertTrue(isinstance(self.working_quant_class.hospital_probHij, np.ndarray))


    def test_getProbablePrimarySchoolsByMSOAIZ(self):
        
        result = self.working_quant_class.getProbablePrimarySchoolsByMSOAIZ()

        self.assertTrue(isinstance(result, list))

    def test_getProbableSecondarySchoolsByMSOAIZ(self):
        
        result = self.working_quant_class.getProbableSecondarySchoolsByMSOAIZ()

        self.assertTrue(isinstance(result, list))

    def test_getProbableRetailByMSOAIZ(self):
        
        result = self.working_quant_class.getProbableRetailByMSOAIZ()

        self.assertTrue(isinstance(result, list))

    def test_getProbableHospitalByMSOAIZ(self):
        
        result =  self.working_quant_class.getProbableHospitalByMSOAIZ()

        self.assertTrue(isinstance(result, list))

    def test_get_flows(self):
        
        result = self.working_quant_class.get_flows(venue = 'PrimarySchool', 
                                                    msoa_list = ['MSOA1','MSOA2','MSOA3'], 
                                                    threshold = 5, 
                                                    thresholdtype = 'nr')

        self.assertTrue(isinstance(result, pd.DataFrame))


if __name__ == '__main__':
    unittest.main()