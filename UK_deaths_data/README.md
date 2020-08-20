## UK Deaths Data
Population data for local authorities was accessed from the ONS [population projections for local authorities](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/datasets/localauthoritiesinenglandtable2).

UK covid-19 death data by local authority was accessed from gov.uk: [Death registrations and occurrences by local authority and health board ](https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/causesofdeath/datasets/deathregistrationsandoccurrencesbylocalauthorityandhealthboard) 

The current _"all_deaths.csv"_ is raw data downloaded from the ONS, however it is very out of date so it is worth replacing this with the most up to date data from the ONS site.

The `clean_data.py` script generates a csv file with cleaned data for all local authorities. This can then be loaded
by the covid19_data.py class, which calculates the start and end date from the data and generates a timeseries for a specific local authority, with the number of deaths on each day starting from the first day in the data.