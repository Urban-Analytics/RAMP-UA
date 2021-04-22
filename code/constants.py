import os

class Constants:
    """Used to reflect the folder structure expected by the code"""    
    
    class Paths:
        AZURE_URL = "https://ramp0storage.blob.core.windows.net/"
        PROJECT_FOLDER_ABSOLUTE_PATH = "" # leave this empty, will inputted from the default.yml file
        # SOURCE_FOLDER = "microsim"
        class CODE:
            FOLDER = "code"
            FULL_PATH = os.path.join(PROJECT_FOLDER_ABSOLUTE_PATH,FOLDER)
        
        # REGIONAL_DATA_FOLDER = "regional_data"
        # COMMON_DATA_FOLDER = "common_data"
        # OUTPUT_FOLDER = "output"
        # CACHE_FOLDER = "cache"
        # TESTS_FOLDER = "tests"
        # DUMMYDATA_FOLDER = "dummy_data"
        LIST_MSOAS_FILE = "model_parameters/test_msoalist.csv" ## better in parameters! (default.yml)
        class DATA:
            FOLDER = "data"
            FULL_PATH_FOLDER = os.path.join(PROJECT_FOLDER_ABSOLUTE_PATH,FOLDER)
        #--> DATA
        class RAW_DATA:
            FOLDER = "raw_data"
            FULL_PATH_FOLDER = os.path.join(DATA.FULL_PATH_FOLDER,FOLDER)
        #-->--> RAW_DATA
        class REFERENCE_DATA:
            FOLDER = "reference_data"
            FULL_PATH_FOLDER = os.path.join(RAW_DATA.FULL_PATH_FOLDER,FOLDER)
        #-->-->--> REFERENCE_DATA
        class LUT:
            FILE = "lookUp.csv"
            FULL_PATH_FILE = os.path.join(REFERENCE_DATA.FULL_PATH_FOLDER,FILE)
        class SEEDING_FILE:
            FILE = "england_initial_casesCTY_tbc.csv"
            FULL_PATH_FILE = os.path.join(REFERENCE_DATA.FULL_PATH_FOLDER,FILE)
        #<--<--<-- REFERENCE_DATA
        #-->-->--> NATIONAL_DATA
        class NATIONAL_DATA:
            FOLDER = "national_data"
            FULL_PATH_FOLDER = os.path.join(RAW_DATA.FULL_PATH_FOLDER,FOLDER)
        class TIME_AT_HOME:
            FILE = "timeAtHomeIncreaseCTY.csv"
        class COMMUTING:
            FILE = "commutingOD.csv"
        #-->-->-->-->QUANT
        class QUANT:
            FOLDER = "QUANT_RAMP"
            FULL_PATH_FOLDER = os.path.join(NATIONAL_DATA.FULL_PATH_FOLDER,FOLDER)
        class PRIMARYSCHOOLS:
            FILE = "primaryZones.csv"
            FULL_PATH_FILE = os.path.join(QUANT.FULL_PATH_FOLDER,FILE)
        class SECONDARYSCHOOLS:
            FILE = "secondaryZones.csv"
            FULL_PATH_FILE = os.path.join(QUANT.FULL_PATH_FOLDER,FILE)
        class RETAIL:
            FILE = "retailpointsZones.csv"
            FULL_PATH_FILE = os.path.join(QUANT.FULL_PATH_FOLDER,FILE)
        #<--<--<--<-- QUANT
        #-->-->-->-->MSOAS_SHAPEFILE
        class MSOAS_SHAPEFILE:
            FOLDER = "MSOAS_shp"
            FULL_PATH_FOLDER = os.path.join(NATIONAL_DATA.FULL_PATH_FOLDER,FOLDER)
        class SHP:
            FILE = "bcc21fa2-48d2-42ca-b7b7-0d978761069f2020412-1-12serld.j1f7i.shp"
            FULL_PATH_FILE = os.path.join(MSOAS_SHAPEFILE.FULL_PATH_FOLDER,FILE)
        #<--<--<--<-- MSOAS_SHAPEFILE
        #<--<--<-- NATIONAL_DATA
        #<--<--RAW_DATA
        #<-- DATA:
        
        # INIT_DATA_MSOAS_RISK = "initial_cases.csv"
        # INIT_DATA_CASES = "msoas.csv"
        class INITIALISATION:
            INITIALISE_FOLDER = "initialise"

        class MODEL:
            MODEL_FOLDER = "model"
            class OPENCL:
                OPENCL_FOLDER = "opencl"
                OPENCL_FONTS_FOLDER = "fonts"
                FONT_DROID = "DroidSans.ttf"
                FONT_ROBOTO = "RobotoMono.ttf"
                OPENCL_SOURCE_FOLDER = "ramp"
                class SOURCE:
                    OPENCL_KERNELS_FOLDER = "kernels"
                    KERNEL_FILE = "ramp_ua.cl"
                    OPENCL_SHADERS_FOLDER = "shaders"
                OPENCL_SNAPSHOTS_FOLDER = "snapshots"
                OPENCL_CACHE_FILE = "cache.npz"



    class Thresholds:
        SCHOOL = 5
        SCHOOL_TYPE = "nr"
        RETAIL = 10
        RETAIL_TYPE = "nr"


class ColumnNames:
    """Used to record standard dataframe column names used throughout"""

    LOCATION_DANGER = "Danger"  # Danger associated with a location
    LOCATION_NAME = "Location_Name"  # Name of a location
    LOCATION_ID = "ID"  # Unique ID for each location

    # # Define the different types of activities/locations that the model can represent
    class Activities:
        RETAIL = "Retail"
        PRIMARY = "PrimarySchool"
        SECONDARY = "SecondarySchool"
        HOME = "Home"
        WORK = "Work"
        ALL = [RETAIL, PRIMARY, SECONDARY, HOME, WORK]

    ACTIVITY_VENUES = "_Venues"  # Venues an individual may visit. Appended to activity type, e.g. 'Retail_Venues'
    ACTIVITY_FLOWS = "_Flows"  # Flows to a venue for an individual. Appended to activity type, e.g. 'Retail_Flows'
    ACTIVITY_RISK = "_Risk"  # Risk associated with a particular activity for each individual. E.g. 'Retail_Risk'
    ACTIVITY_DURATION = "_Duration" # Column to record proportion of the day that individuals do the activity
    ACTIVITY_DURATION_INITIAL = "_Duration_Initial"  # Amount of time on the activity at the start (might change)

    # Standard columns for time spent travelling in different modes
    TRAVEL_CAR = "Car"
    TRAVEL_BUS = "Bus"
    TRAVEL_TRAIN = "Train"
    TRAVEL_WALK = "Walk"

    INDIVIDUAL_AGE = "DC1117EW_C_AGE" # Age column in the table of individuals
    INDIVIDUAL_SEX = "DC1117EW_C_SEX"  # Sex column in the table of individuals
    INDIVIDUAL_ETH = "DC2101EW_C_ETHPUK11"  # Ethnicity column in the table of individuals

    # Columns for information about the disease. These are needed for estimating the disease status

    # Disease status is one of the following:
    class DiseaseStatuses:
        SUSCEPTIBLE = 0
        EXPOSED = 1
        PRESYMPTOMATIC = 2
        SYMPTOMATIC = 3
        ASYMPTOMATIC = 4
        RECOVERED = 5
        DEAD = 6
        ALL = [SUSCEPTIBLE, EXPOSED, PRESYMPTOMATIC, SYMPTOMATIC, ASYMPTOMATIC, RECOVERED, DEAD]
        assert len(ALL) == 7

    DISEASE_STATUS = "disease_status"  # Which one it is
    DISEASE_STATUS_CHANGED = "status_changed"  # Whether it has changed between the current iteration and the last
    DISEASE_PRESYMP = "presymp_days"
    DISEASE_SYMP_DAYS = "symp_days"
    DISEASE_EXPOSED_DAYS = "exposed_days"

    #DAYS_WITH_STATUS = "Days_With_Status"  # The number of days that have elapsed with this status
    CURRENT_RISK = "current_risk"  # This is the risk that people get when visiting locations.

    # No longer update disease counts per MSOA etc. Not needed
    MSOA_CASES = "MSOA_Cases"  # The number of cases per MSOA
    HID_CASES = "HID_Cases"  # The number of cases in the individual's house