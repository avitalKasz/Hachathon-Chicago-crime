import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import os
from folium.plugins import HeatMap


def load_data_sets():
    # crimes_full = pd.read_csv("crimes_full.csv")
    df = pd.read_csv("dataset_crimes.csv")
    # differing_feature = {"BATTERY", "ASSAULT", "THEFT", "CRIMINAL "
    #                                                          "DAMAGE",
    #                           "DECEPTIVE PRACTICE"}
    # crimes_full = crimes_full[(crimes_full["Primary Type"] == "BATTERY") |
    #                           (crimes_full["Primary Type"] == "ASSAULT") |
    #                           (crimes_full["Primary Type"] == "THEFT") |
    #                           (crimes_full["Primary Type"] == "DECEPTIVE PRACTICE") |
    #                           (crimes_full["Primary Type"] == "CRIMINAL DAMAGE")]
    df["Date"] = pd.to_datetime(df["Date"])
    df['crime_day'] = df["Date"].dt.dayofweek
    return df


# def check_if_in_set(crimes_full):
#     for

def create_heatmap(df_acc, type):

    heat_df = clean_geo_data(df_acc)
    hmap1 = folium.Map(location=[41.881832, -87.623177], zoom_start=10, )

    # List comprehension to make out list of lists
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in
                 heat_df.iterrows()]
    #
    # for index, row in df_acc.iterrows():
    #     folium.CircleMarker([row['Latitude'], row['Longitude']],
    #                         radius=15,
    #                         fill_color="#3db7e4",  # divvy color
    #                         ).add_to(df_acc)

    # Plot it on the map
    HeatMap(heat_data, blur=10, radius=10, min_opacity=0.1).add_to(hmap1)
    title_html = '''
                 <h3 align="center" style="font-size:16px"><b>Weekday = {}</b></h3>
                 '''.format(type)
    # Display the map
    hmap1.get_root().html.add_child(folium.Element(title_html))

    return hmap1


def clean_geo_data(df_acc):
    df_acc['Latitude'] = df_acc['Latitude'].astype(float)
    df_acc['Longitude'] = df_acc['Longitude'].astype(float)
    heat_df = df_acc[['Latitude', 'Longitude']]

    heat_df = heat_df.dropna(axis=0, subset=['Latitude', 'Longitude'])
    return heat_df


def wrap_heat_map(path, df, type):
    if path in os.listdir():
        os.remove(path)
    hmap1 = create_heatmap(df, type)
    hmap1.save(path)

def gen_map_for_each(differing_feature, feature_name):
    df = load_data_sets()
    for type in differing_feature:
        current_df = df[df[feature_name] == type].copy()
        wrap_heat_map("days_heat_map\{}_map.html".format(type), current_df,
                      type)




if __name__ == '__main__':
    # df_acc = load_data_sets()
    # df_acc['Latitude'] = df_acc['Latitude'].astype(float)
    # df_acc['Longitude'] = df_acc['Longitude'].astype(float)
    # gdf = gpd.GeoDataFrame(
    #     df, geometry=gpd.points_from_xy(df.Latitude, df.Longitude))
    # gen_map_for_each({"BATTERY", "ASSAULT", "THEFT", "CRIMINAL "
    #                                                          "DAMAGE",
    #                           "DECEPTIVE PRACTICE"}, "Primary Type")
    gen_map_for_each({0,1,2,3,4,5,6}, "crime_day")
