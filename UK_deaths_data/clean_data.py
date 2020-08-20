import pandas as pd
from datetime import *
from dateutil.relativedelta import *


def clean_population_data():
    population_df = pd.read_csv("./local_authority_population.csv")
    population_df.rename(columns={"2018": "population"})
    population_df = population_df[population_df["AGE GROUP"] == "All ages"]
    population_df = population_df.filter(['CODE', "AREA", "2018"])
    population_df = population_df.rename(columns={"2018": "POPULATION"})
    population_df['POPULATION'] = population_df['POPULATION'].str.replace(',', '')
    population_df['POPULATION'] = pd.to_numeric(population_df['POPULATION'])
    population_df = population_df.sort_values(by=['POPULATION'])
    print("writing cleaned population data to file")
    population_df.to_csv("./local_authority_population_cleaned.csv", index=False)


def clean_deaths_data():
    deaths_df = pd.read_csv("./all_deaths.csv")
    deaths_df = deaths_df[deaths_df["Cause of death"] == "COVID 19"]
    deaths_df = deaths_df.filter(['Area code', "Area name", "Week number", "Number of deaths"])
    grouped = deaths_df.groupby(["Area code", "Area name", "Week number"])['Number of deaths'].sum()
    deaths_df = pd.DataFrame(grouped)
    deaths_df = deaths_df.reset_index()
    deaths_df["date"] = deaths_df["Week number"].apply(get_week_end_date)
    print("writing cleaned deaths data to file")
    deaths_df.to_csv("./coronavirus-deaths_cleaned.csv", index=False)


def get_week_end_date(week_number):
    start_date = datetime(2020, 1, 3)  # end of first week
    week_end_date = start_date + relativedelta(weeks=week_number-1)
    return week_end_date.strftime("%Y-%m-%d")


def main():
    clean_population_data()
    clean_deaths_data()


if __name__ == '__main__':
    main()
