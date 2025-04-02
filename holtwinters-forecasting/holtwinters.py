from estimator import *
import os
import pandas as pd

class HoltWintersEstimator(Estimator):
    """ A class that implements the Holt-Winters model for estimation. 
    Attributes:
        number_of_seasons: Number of seasons in the time series. If you have 3 years of observations, with month by month data, number_of_seasons=3.
        season_size: Indicates the number of observations in each season. For example, if you have 3 years of observation, with month by month data, season_size=12, which is the number of months in a year.
        seasonal_factor: Indicates the seasonal factor of each observation on your data set.
        average_component: Indicates the average component in the Holt-Winters model.
        tendence_component: Indicates the tendence component in the Holt-Winters model.
        seasonal_component: Indicates the seasonal component in the Holt-Winters model. It is a matrix whose first index is the seasonal index of a given time, and the second index is the season itself. 
    """

    def __init__(self, time_series, number_of_seasons, season_size):
        """ Initializes a Holt-Winters estimator. """
        super(HoltWintersEstimator, self).setTimeSeries(time_series)
        self.number_of_seasons = number_of_seasons
        self.season_size = season_size
        self.seasonal_factor = []
        self.average_component = []
        self.tendence_component = []
        self.seasonal_component = []
        self.time_series = time_series
        self.initializeComponents()
        self.calculateComponents(0.2, 0.15, 0.05)

        # Ensure the components are long enough for prediction
        while len(self.average_component) < len(time_series) + 12:
            self.average_component.append(self.average_component[-1])
            self.tendence_component.append(self.tendence_component[-1])

    def seasonOfTime(self, time):
        if (time % self.season_size) == 0:  # 修改这里
            return int((time / self.season_size))
        return int((time / self.season_size)) + 1

    def seasonalIndexOfTime(self, time):
        seasonal_index = time % (self.season_size)
        if seasonal_index == 0:  # 修改这里
            seasonal_index = self.season_size
        return seasonal_index

    def seasonMovingAverage(self, season):
        """ Returns the moving average of a season. """
        floor = (season - 1) * self.season_size
        ceil = season * self.season_size
        moving_average = 0.0

        for y in self.time_series[floor:ceil]:
            moving_average += y
        return moving_average/self.season_size
    
    def seasonalFactor(self, time):
        """ Calculates the seasonal factor for each historical data in the time series."""
        season = self.seasonOfTime(time)
        moving_average = self.seasonMovingAverage(season)
        seasonal_index = self.seasonalIndexOfTime(time)
        tendence_component = self.tendence_component[0]

        tendence = (((self.season_size + 1.0) / 2.0) - seasonal_index) * tendence_component
        seasonal_index_average = moving_average - tendence

        # Prevent division by zero
        if seasonal_index_average == 0:
            seasonal_index_average = 1e-10

        factor = (self.time_series[time - 1] / seasonal_index_average)

        return factor

    def insertSeasonalComponent(self, seasonal_index, season, value):
        """ Inserts a value in the seasonal component matrix.
        """
        if (len(self.seasonal_component)) < seasonal_index:
            while len(self.seasonal_component) < seasonal_index:
                self.seasonal_component.append([])
        
        if len(self.seasonal_component[seasonal_index - 1]) < (season + 1):
            while len(self.seasonal_component[seasonal_index - 1]) < (season + 1):
                self.seasonal_component[seasonal_index - 1].append(None)
        self.seasonal_component[seasonal_index - 1][season] = value

    def getSeasonalComponent(self, time):
        """ Returns a value from the seasonal component matrix.
        """
        if time <= 0:
            season = 0
            seasonal_index = self.seasonalIndexOfTime(time + self.season_size)
        else:
            seasonal_index = self.seasonalIndexOfTime(time)
            season = self.seasonOfTime(time)
        component = self.seasonal_component[seasonal_index - 1][season]
        if component == 0:
            component = 1e-10  # Prevent division by zero
        return component

    def estimate(self, time, base_time):
        """ Estimate the value of the function on time, based on the base_time observation. Often base_time is time-1. """
        if base_time >= len(self.average_component) or base_time >= len(self.tendence_component):
            raise IndexError("Base time index is out of range for the components.")
        estimation = (self.average_component[base_time] + self.tendence_component[base_time] * (time - base_time)) * self.getSeasonalComponent(time - self.season_size)
        return estimation

    def initializeComponents(self):
        """ Initializes the components of the Holt-Winters model. """
        first_season_average = self.seasonMovingAverage(1)
        last_season_average = self.seasonMovingAverage(self.number_of_seasons)

        #tendence component
        self.tendence_component.append((last_season_average - first_season_average) / ((self.number_of_seasons - 1) * self.season_size))
        
        #average component
        self.average_component.append( first_season_average - ((self.season_size / 2.0) * self.tendence_component[0]))
        
        #seasonal component
        for time in range(1, len(self.time_series) + 1):
            self.seasonal_factor.append(self.seasonalFactor(time))

        # Ensure self.seasonal_factor is long enough before accessing indices
        while len(self.seasonal_factor) < (self.number_of_seasons * self.season_size):
            self.seasonal_factor.append(1.0)  # Default value to prevent out of range error
        
        seasonal_index_average = []
        for seasonal_index in range(1, self.season_size + 1):
            seasonal_index_sum = 0.0
            for m in range(self.number_of_seasons):
                index = seasonal_index + (m * self.season_size)
                factor = self.seasonal_factor[index - 1]
                seasonal_index_sum += factor
            seasonal_index_average.append(seasonal_index_sum *  (1.0 / self.number_of_seasons))

        snt_average_sum = 0.0
        for snt_average in seasonal_index_average:
            snt_average_sum += snt_average

        # Prevent division by zero
        if snt_average_sum == 0:
            snt_average_sum = 1e-10
        
        adjustment_level = self.season_size / snt_average_sum

        for seasonal_index in range(1, self.season_size + 1):
            value = seasonal_index_average[seasonal_index - 1] * adjustment_level
            self.insertSeasonalComponent(seasonal_index, 0, value)

    def calculateComponents(self, alpha, beta, gamma):
        for time in range(1, len(self.time_series) + 1):
            seasonal_component = self.getSeasonalComponent(time - self.season_size)
            if seasonal_component == 0:
                seasonal_component = 1e-10  # Prevent division by zero
            average_component = (alpha * (self.time_series[time - 1] / seasonal_component)) + ((1 - alpha) * (self.average_component[time - 1] + self.tendence_component[time - 1]))
            self.average_component.append(average_component)

            tendence_component = (beta * (self.average_component[time] - self.average_component[time - 1])) + ((1 - beta) * self.tendence_component[time - 1])
            self.tendence_component.append(tendence_component)

            if self.average_component[time] == 0:
                self.average_component[time] = 1e-10  # Prevent division by zero
            seasonal_component = gamma * self.time_series[time - 1] / self.average_component[time] + (1 - gamma) * self.getSeasonalComponent(time - self.season_size)
            index = self.seasonalIndexOfTime(time)
            season = self.seasonOfTime(time)
            self.insertSeasonalComponent(index, season, seasonal_component)

# 读取文件夹中的所有CSV文件
input_folder = "/home/sunyongqian/liuheng/shenchao/kontrast/dataset/data/aiops2024"
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        input_file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(input_file_path)

        # 假设CSV文件有'timestamp'和'value'两列
        time_series = df['value'].tolist()

        # 创建HoltWintersEstimator对象
        hwe = HoltWintersEstimator(time_series, 3, 12)

        # 打开输出文件
        output_file_path = os.path.join(output_folder, file_name)
        with open(output_file_path, "w") as output_file:
            # 预测未来的值
            for t in range(len(time_series) + 1, len(time_series) + 13):
                try:
                    predicted_value = hwe.estimate(t, t - 1)
                    output_file.write(f"Predicted value at time {t}: {predicted_value}\n")
                except IndexError as e:
                    output_file.write(f"Error predicting value at time {t}: {e}\n")

print(f"Predictions saved to {output_folder}")