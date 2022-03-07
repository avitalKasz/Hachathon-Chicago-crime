# Hackathon_IML

class neighbors_crimes.py  - this class trains the crime location.
load - this function loads the given data
clean_geo_data - this function cleans the geographic data
get_neighbors_crimes - trains the data on the crime locations

class etl - this class receives data, cleans it and the trains the given data.

class enrich_df - this class joins data's.
join_enrich - this functions joins our data with the external data inorder to
enrich our data.

class classifier - this class creates an object of CrimePredictor
predict - this function predicts at witch location there will be a crime.class
send_police_cars - this function sens the police cars to the location were
we predict there will be a crime.

