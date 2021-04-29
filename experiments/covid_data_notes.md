# Notes about COVID data and RAMP seeding

At some point these should be integrated into the documentation properly

## Current seeding procedure

@spoonerf please could you add a few notes about how the seeding is done currently?

## Goals

The user would have to provide the number of initial cases per studied MSOA and a starting date as inputs.

User would be given:
* a csv with cases at MSOA level as close to true infection/symptomatic rates as possible for reference.
* a list of theoretical MSOA risks to prepare data for 'what if' scenarios for reference (already exists).

Code would have two options:
* Use exactly the data provided (at MSOA level)
* Use data provided as probability weights to introduce some randomness.

## Data needed | Status

* __Daily/weekly reported cases at MSOA level__ | weekly since 5th March 2020 at MSOA level for all of England: [gov](https://api.coronavirus.data.gov.uk/v2/data?areaType=msoa&metric=newCasesBySpecimenDateRollingSum&format=csv).
* __Cases to infections__ | Check [link](https://www.medrxiv.org/content/medrxiv/early/2020/04/17/2020.04.13.20062760.full.pdf) for guidance?
* and/or __Cases to symptomatic__
* __Alt: daily/weekly deaths__
* __Alt: Deaths to infections__
* and /or __Alt: Deaths to symptomatic__
* __Sick population profile to distribute cases within population__ | By age [gov](https://coronavirus.data.gov.uk/downloads/demographic/cases/specimenDate_ageDemographic-unstacked.csv) (haven't checked content yet). See how it's done currently and if there's need for improvement or not?

## More recent APIs

[coronavirus.data.gov.uk](https://coronavirus.data.gov.uk/details/download) now provides access to quite detailed COVID test data.

The website allows for the creation of URLs that can be used to download the data automatically.

E.g. [this url](https://api.coronavirus.data.gov.uk/v2/data?areaType=msoa&areaCode=E08000035&metric=newCasesBySpecimenDateRollingSum&metric=newCasesBySpecimenDateRollingRate&format=csv) will provide a CSV file with new cacses (`newCasesBySpecimenDateRollingRate`) at by MSOAs in Leeds starting in April 2020:

| regionCode | regionName               | UtlaCode  | UtlaName | LtlaCode  | LtlaName | areaCode  | areaName    | areaType | date       | newCasesBySpecimenDateRollingRate | newCasesBySpecimenDateRollingSum |
| ---------- | ------------------------ | --------- | -------- | --------- | -------- | --------- | ----------- | -------- | ---------- | --------------------------------- | -------------------------------- |
| E12000003  | Yorkshire and The Humber | E08000035 | Leeds    | E08000035 | Leeds    | E02002332 | Otley North | msoa     | 22/04/2021 | 48.8                              | 3                                |
| E12000003  | Yorkshire and The Humber | E08000035 | Leeds    | E08000035 | Leeds    | E02002333 | Otley South | msoa     | 22/04/2021 | 39.3                              | 3                                |


## Questions

 - Can we replace the original seeding data with more recent data from [coronavirus.data.gov.uk](https://coronavirus.data.gov.uk/details/download)? 

 - If not (maybe we need data earlier than April 2020) can we use these new data for seeding more recent scenarios?

