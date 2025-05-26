import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import unicodedata


class CoordinateLoader:
    def __init__(self):
        project_path = os.path.dirname(os.path.realpath(__file__))
        self.geojson_file = os.path.join(project_path,
                                         "../data/countries.geojson")
        self.african_list_file = os.path.join(project_path,
                                              "../data/african_countries.json")
        self.ws_file = os.path.join(project_path,
                                    "../data/western_sahara.json")

        self.name_mapping = {
            "Cape Verde": "Cabo Verde",
            "Congo": "Republic of the Congo",
            "DR Congo": "Democratic Republic of the Congo",
            "Swaziland": "eSwatini",  # Match raw GeoJSON spelling
            "Sao Tome and Principe": "São Tomé and Principe"
        }

        self.africa_map = gpd.read_file(self.geojson_file)
        self.african_countries_target = self._load_african_country_list()

        def normalize(name):
            return unicodedata.normalize('NFKD', name).encode('ASCII',
                                                              'ignore').decode().lower()

        # Build mapping from normalized name to actual GeoJSON name
        normalized_geojson_names = {
            normalize(name): name for name in self.africa_map["name"].dropna().unique()
        }

        # Match African target list to actual GeoJSON names
        african_names_in_geojson = [
            normalized_geojson_names[name] for name in self.african_countries_target
            if name in normalized_geojson_names
        ]

        self.africa_map = self.africa_map[
            self.africa_map["name"].isin(african_names_in_geojson)
        ].copy()

        if "Western Sahara" not in self.africa_map["name"].values:
            self._add_western_sahara_from_file()

        self.coordinates = self._load_coordinates()

    def _load_african_country_list(self):
        with open(self.african_list_file, "r", encoding="utf-8") as f:
            return set(json.load(f))

    def _add_western_sahara_from_file(self):
        with open(self.ws_file, "r", encoding="utf-8") as f:
            ws_data = json.load(f)

        polygon = Polygon(ws_data["coordinates"])

        new_row = gpd.GeoDataFrame([{
            "name": ws_data["name"],
            "geometry": polygon
        }], crs=self.africa_map.crs)

        self.africa_map = pd.concat([self.africa_map, new_row], ignore_index=True)

    def _load_coordinates(self):
        coords = {}
        self.africa_map['centroid'] = self.africa_map.geometry.centroid
        self.africa_map['lon'] = self.africa_map.centroid.x
        self.africa_map['lat'] = self.africa_map.centroid.y

        for idx, row in self.africa_map.iterrows():
            country_name = row['name'].strip()
            coords[country_name] = (row['lat'], row['lon'])

        return coords

    def get_coordinates(self, country):
        country_key = self.name_mapping.get(country.strip(), country.strip())
        return self.coordinates.get(country_key)

    def get_all_coordinates(self):
        return self.coordinates
