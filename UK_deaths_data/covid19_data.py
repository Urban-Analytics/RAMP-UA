import pandas as pd
import matplotlib.pyplot as plt


class Covid19Data:
    def __init__(self, data_path, area):
        self.area = area

        self.data_path = data_path
        self.deaths_data_path = data_path + "/coronavirus-deaths_cleaned.csv"
        self.population_data_path = data_path + "/local_authority_population_cleaned.csv"

        self.start_date = None
        self.end_date = None

        self.population_data = self.load_population_data()
        self.deaths = self.load_deaths_data()

        self.deaths_timeseries = self.init_timeseries()
        self.num_timesteps = len(self.deaths)

    def load_deaths_data(self):
        deaths_df = pd.read_csv(self.deaths_data_path)
        deaths_df['date'] = pd.to_datetime(deaths_df['date'], format='%Y-%m-%d')
        area_df = get_col_for_area(deaths_df, self.area, "Number of deaths")
        return area_df

    def load_population_data(self):
        population_df = pd.read_csv(self.population_data_path, index_col="AREA")
        population_df['POPULATION'] = population_df['POPULATION'].astype(int)
        return population_df.to_dict()['POPULATION']

    def get_population(self):
        return self.population_data[self.area]

    def init_timeseries(self):
        """
        get concatenated timeseries of data for calibration
        """

        self.start_date = self.deaths['date'].min()
        self.end_date = self.deaths['date'].max()

        self.deaths['Cumulative deaths'] = self.deaths["Number of deaths"].cumsum()
        print("loaded data for time period: start: {}, end: {}".format(self.start_date, self.end_date))

        return self.deaths['Cumulative deaths'].to_numpy()


def get_col_for_area(df, area_name, col_name):
    area_groups = df.groupby("Area name")
    area_df = area_groups.get_group(area_name)
    return area_df.filter(["date", col_name])


def main():
    covid19_data = Covid19Data("./", area="Westminster")
    plt.plot(covid19_data.deaths_timeseries)
    plt.show()


if __name__ == '__main__':
    main()
