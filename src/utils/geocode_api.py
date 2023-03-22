from dotenv import dotenv_values
import json
import googlemaps
import datetime


# config = dotenv_values("../../.env")
# API_KEY = config["MAPS_API_KEY"]

# gmaps = googlemaps.Client(key=API_KEY)

# Geocoding an address
# city_coord = {"lat": [], "lng": []}
# for city in df["HomeTeam_City"].unique():
#     geocode_result = gmaps.geocode(city)
#     res = geocode_result[0]["geometry"]["location"]
#     city_coord["lat"].append(res["lat"])
#     city_coord["lng"].append(res["lng"])

# city_coord = pd.DataFrame(city_coord, index=np.arange(len(city_coord["lat"])))

# city_coord = pd.concat([pd.Series(df["HomeTeam_City"].unique()), city_coord], axis=1)

# city_coord.to_csv(DATA_PATH.joinpath("raw", "city_coords.csv"))