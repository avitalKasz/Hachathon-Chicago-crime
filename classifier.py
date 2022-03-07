

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


def predict(X):
    pass


def send_police_cars(X):
    pass


class CrimePredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        self.weather_data = pd.read_csv(path_to_weather, na_values = "None")

        self.catboost_reg_model = joblib.load("cb_model.pkl")
        self.gbm_classifier = joblib.load("gbm_model.pkl")
        self.one_hot_encoder = joblib.load("one_hot_encoder.pkl")

    def predict(self, x):
        """
        Receives a pandas DataFrame of shape (m, 15) with m flight features,
        and predicts their delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        x = x.drop(columns=['Tail_Number', "Flight_Number_Reporting_Airline"])
        x["in_state_flight"] = x['OriginState'] == x['DestState']
        x = x.drop(columns=["OriginState", "DestState", "OriginCityName", "DestCityName"])
        self.weather_data.day = pd.to_datetime(self.weather_data.day)
        x.FlightDate = pd.to_datetime(x.FlightDate)
        x['day'] = x.FlightDate.dt.day
        x['month'] = x.FlightDate.dt.month
        x['year'] = x.FlightDate.dt.year
        zibimputer = SimpleImputer(strategy='constant', fill_value=0)
        zibimputer.fit(self.weather_data[['snow_in', 'snowd_in', 'precip_in']])
        self.weather_data[['snow_in', 'snowd_in', 'precip_in']] = zibimputer.prepare_X(
            self.weather_data[['snow_in', 'snowd_in', 'precip_in']])
        self.weather_data.rename(columns={"day": 'date'}, inplace=True)
        per_day_origin = x.groupby(['year', 'month', 'day', 'Origin']).count()[
            'DayOfWeek'].reset_index()
        per_day_dest = x.groupby(['year', 'month', 'day', 'Dest']).count()['DayOfWeek'].reset_index()
        out_amount = x[['year', "month", 'day']].drop_duplicates().merge(per_day_origin,
                                                                         on=['year', 'month', 'day'],
                                                                         how="outer").rename(
            columns={"DayOfWeek": "outgoing_amount"})
        in_amount = x[['year', "month", 'day']].drop_duplicates().merge(per_day_dest,
                                                                        on=['year', 'month', 'day'],
                                                                        how="outer").rename(
            columns={"DayOfWeek": "incoming_amount"})
        x = x.merge(out_amount, on=['year', 'month', 'day', 'Origin'], how='left').merge(
            in_amount, on=['year', 'month', 'day', 'Dest'], how='left')

        x["adj_incoming"] = (x["incoming_amount"] - x["incoming_amount"].mean()) / \
                            x["incoming_amount"].std()
        x["adj_outgoing"] = (x["outgoing_amount"] - x["outgoing_amount"].mean()) / \
                            x["outgoing_amount"].std()
        x["adj_incoming"]=x["adj_incoming"].fillna(0)
        x["adj_outgoing"] = x["adj_outgoing"].fillna(0)
        x['sin_time'] = np.sin(2 * np.pi * (x['CRSDepTime'] * 60 * 60) / (24 * 60 * 60))
        x['cos_time'] = np.cos(2 * np.pi * (x['CRSDepTime'] * 60 * 60) / (24 * 60 * 60))
        x.drop(columns=['CRSDepTime', 'CRSArrTime'], inplace=True)

        x = x.merge(self.weather_data, left_on=["FlightDate", "Origin"],
                    right_on=["date", 'station'], how='left', )
        x = x.merge(self.weather_data, left_on=["FlightDate", "Dest"],
                    right_on=["date", 'station'], how='left', suffixes=('_origin', '_dest'))

        x = x.drop(columns=['FlightDate','station_origin', 'station_dest','date_dest', 'date_origin'])

        x['pred'] = self.catboost_reg_model.predict(x)
        x = x[list(x.columns[:2]) + ['pred'] + [i for i in x.columns[2:] if i != 'pred']]

        encoded = self.one_hot_encoder.prepare_X(x[["DayOfWeek", "Reporting_Airline", "Origin", "Dest", 'day', 'month',
                                                    'year']])
        post_dum = pd.merge(x.reset_index().drop(columns=['index']),
                            pd.DataFrame.sparse.from_spmatrix(encoded), left_index=True, right_index=True)
        post_dum.drop(columns=["DayOfWeek", "Reporting_Airline", "Origin", "Dest", 'day', 'month',
                               'year'],inplace=True)

        post_dum['class'] = self.gbm_classifier.predict(post_dum)

        result = post_dum[['pred', 'class']]
        result['class'] = result['class'].apply(lambda x: classes[x])
        result['pred'] = result['class'].apply(lambda x:max([x,15]))
        return result