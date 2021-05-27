# Data to run the microsim model on different regions.

*TO BE UPDATED !!*


The data folder structure currently has a 'common data' folder (data at national level) and can have several regional data folders which are named according to the user.

The national data folder contains:
- `MSOAS_shp` (folder): shapefile with MSOAs (... @HSalat can help to describe what this file is)
- `QUANT_RAMP` (folder): tables (csv files) originated by the QUANT model

Each regional folder contains the following:
- osm (folder): contains the `gis_osm_buildings_a_free_1` shapefile for the specific region
- `commuting_od.csv`
- `google_mobility_lockdown_daily_14_day_moving_average.csv`
- `initial_cases.csv`
- `msoas_risk.csv`
- `simulated_TU_keyworker_health.csv`


The data are stored using [Git Large File Storage (LFS)](https://git-lfs.github.com/)
(redundant?)


