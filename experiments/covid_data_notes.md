# Notes about COVID data and RAMP seeding

At some point these should be integrated into the documentation properly

## Current seeding procedure

@spoonerf please could you add a few notes about how the seeding is done currently?

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

