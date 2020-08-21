class ColumnNames:
    """Used to record standard dataframe column names used throughout"""

    LOCATION_DANGER = "Danger"  # Danger associated with a location
    LOCATION_NAME = "Location_Name"  # Name of a location
    LOCATION_ID = "ID"  # Unique ID for each location

    # Define the different types of activities/locations that the model can represent
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
    ACTIVITY_DURATION = "_Duration" # Column to record proportion of the day that invividuals do the activity
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
    #MSOA_CASES = "MSOA_Cases"  # The number of cases per MSOA
    #HID_CASES = "HID_Cases"  # The number of cases in the individual's house