from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

train_directory = Path("../data/fars_train.csv")
test_directory = Path("../data/fars_test.csv")


def binary_cat(row_value, zero_value):
    return 0 if row_value == zero_value else 1

# Load data from Dataframe
df = pd.read_csv(train_directory, quoting=1, delimiter=',')

# Clean up Categorical Data
features = ['fatals', 'permvit', 'age', 'deaths', 'month', 'pernotmvit', 've_total']


cleaned_df = df[features]
#cleaned_df['numoccs'] = cleaned_df['numoccs'].fillna(0.0)

#Create columns for Binary Features
cleaned_df['is_ped_fatality'] = df['a_ped_f'].apply(lambda x: binary_cat(x, 'Other Crash'))
cleaned_df['is_weekend'] = df['a_dow_type'].apply(lambda x: binary_cat(x, 'Weekday'))
cleaned_df['is_night'] = df['a_tod_type'].apply(lambda x: binary_cat(x, 'Daytime'))
cleaned_df['is_urban'] = df['a_ru'].apply(lambda x: binary_cat(x, 'Rural'))
cleaned_df['is_inter'] = df['a_inter'].apply(lambda x: binary_cat(x, 'Non-Interstate'))
cleaned_df['is_intsec'] = df['a_intsec'].apply(lambda x: binary_cat(x, 'Non-Intersection'))
cleaned_df['on_roadway'] = df['a_relrd'].apply(lambda x: 1 if x.startswith('On') else 0)
cleaned_df['off_roadway'] = df['a_relrd'].apply(lambda x: 1 if x.startswith('Off') else 0)
cleaned_df['is_junc'] = df['a_junc'].apply(lambda x: 1 if str(x).startswith('Junct') else 0)
cleaned_df['not_junc'] = df['a_junc'].apply(lambda x: 1 if str(x).startswith('Non') else 0)
#cleaned_df['is_bike_fatality'] = df['a_ped_f'].apply(lambda x: binary_cat(x, 'Other Crash'))
#cleaned_df['is_rollover'] = df['a_roll'].apply(lambda x: binary_cat(x, 'Other Crash'))   #hurt multi-nomial output
cleaned_df['is_hit_and_run'] = df['a_hr'].apply(lambda x: binary_cat(x, 'No - Hit and Run'))  #no impact to multi-nomial
cleaned_df['is_police_pursuit'] = df['a_polpur'].apply(lambda x: binary_cat(x, 'Other Crash'))  
cleaned_df['is_ped'] = df['a_ped'].apply(lambda x: binary_cat(x, 'no'))  

#Create columns for non-Ordinal Features
label_encoder = LabelEncoder()     #Create Label Encoder

#cleaned_df['owner_reg'] = label_encoder.fit_transform(df['owner'])
#cleaned_df['impact_loc'] = label_encoder.fit_transform(df['impact1'])
#cleaned_df['weather_cond'] = label_encoder.fit_transform(df['weather'])
#cleaned_df['light_cond'] = label_encoder.fit_transform(df['lgt_cond'])


cleaned_df['body_type'] = df['a_body']
cleaned_df['owner_reg'] = df['owner']
cleaned_df['impact_loc'] = df['impact1']
cleaned_df['weather_cond'] = df['weather']
cleaned_df['light_cond'] = df['lgt_cond']

for col in cleaned_df.dtypes[cleaned_df.dtypes == 'object'].index:
    for_dummy = cleaned_df.pop(col)
    cleaned_df = pd.concat([cleaned_df, pd.get_dummies(for_dummy, prefix=col)], axis=1)


#Create column for Ordinal Features (treating as non-ordinal FOR NOW)
cleaned_df['road_type'] = label_encoder.fit_transform(df['a_roadfc'])
cleaned_df['deform_type'] = label_encoder.fit_transform(df['deformed'])
#cleaned_df['state_occ'] = label_encoder.fit_transform(df['state'])  #very minor neg impat to multi-nomial


preprocessed_data = (cleaned_df, df["driver_factor"])