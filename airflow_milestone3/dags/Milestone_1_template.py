#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def ms1():
  df = pd.read_csv('./data/2018_Accidents_UK.csv', index_col= 'accident_index')
  df.info()
  df.describe()
  df.head()
  plt.figure(figsize=(7,5))
  bar = sns.countplot(x=df["day_of_week"], data= df
                      ,order = df["day_of_week"].value_counts().index)
  plt.title("Accidents per Weekday", fontsize="25")
  plt.xlabel("Weekday", weight="bold", fontsize="15")
  plt.ylabel("Accidents", weight="bold", fontsize="15")
  plt.show()
  print("-Most accidents happen on Friday which might be due to people rushing home for the weekend or due to more people going out with friends/family. \n\n\
  -Least accidents happen on Sunday which might be because people prefer to stay home that day, or \n\
  because a lot of shops close that day or have reduced hours which means less traffic")
  new_df = df[df['accident_severity'] == 'Fatal'].groupby("road_surface_conditions")['accident_severity'].count().astype(float)
  new_df1 = df.groupby("road_surface_conditions")['accident_severity'].count().astype("float")
  for i in range(0,new_df.size):
    num=(new_df[i]*100)/(new_df1[i])
    new_df1[i]=num.round(2)
  new_df1.plot.barh(xlabel="Road Surface Condition")
  df.groupby(by="speed_limit")["number_of_casualties"].mean().plot.line(xlabel='Speed Limit',ylabel='Number of Casualties')
  u=df.number_of_casualties[df['urban_or_rural_area'] == 'Urban'].mean()
  r=df.number_of_casualties[df['urban_or_rural_area'] == 'Rural'].mean()
  if(u>r):
    print("Urban areas have more casualties")
  else:
    print("Rural areas have more casualties")

  plt.title('Average Number of casualties per Accident')
  plt.bar(["Urban","Rural"],[u,r])
  sns.scatterplot(y='number_of_casualties',x='speed_limit',data=df)
  ((df.isnull() | df.isna()).sum() * 100 / df.index.size).round(2)
  df.query("second_road_number.isna()").head()
  modes = df.query("weather_conditions.isna()").mode().iloc()[0]
  weather_conditions_df = df.query("weather_conditions.isna()")

  for i in range(0,len(weather_conditions_df.columns)):
    print("")
    if(weather_conditions_df[weather_conditions_df.columns[i]].dtype == object):
        print(weather_conditions_df.columns[i],"   ",modes[i],"   ",(len(weather_conditions_df.query("{}=='{}'".format(weather_conditions_df.columns[i],modes[i])))/len(weather_conditions_df[weather_conditions_df.columns[i]]))*100,"%")  
    else:
        print(weather_conditions_df.columns[i],"   ",modes[i],"   ",(len(weather_conditions_df.query("{}=={}".format(weather_conditions_df.columns[i],modes[i])))/len(weather_conditions_df[weather_conditions_df.columns[i]]))*100,"%")
  
  modes = df.query("road_type.isna()").mode().iloc()[0]
  road_type_df = df.query("road_type.isna()")

  for i in range(0,len(road_type_df.columns)):
    print("")
    if(road_type_df[road_type_df.columns[i]].dtype == object):
              print(road_type_df.columns[i],"   ",modes[i],"   ",(len(road_type_df.query("{}=='{}'".format(road_type_df.columns[i],modes[i])))/len(road_type_df[road_type_df.columns[i]]))*100,"%")
    else:
        print(road_type_df.columns[i],"   ",modes[i],"   ",(len(road_type_df.query("{}=={}".format(road_type_df.columns[i],modes[i])))/len(road_type_df[road_type_df.columns[i]]))*100,"%")


  modes = df.query("longitude.isna()").mode().iloc()[0]
  location_df = df.query("longitude.isna()")

  for i in range(0,len(location_df.columns)):
    print("")
    if(location_df.columns[i] == 'location_easting_osgr' or location_df.columns[i] == 'latitude' or location_df.columns[i] == 'longitude' or location_df.columns[i] == 'location_northing_osgr'):
      continue
    if(location_df[location_df.columns[i]].dtype == object):
      print(location_df.columns[i],"   ",modes[i],"   ",(len(location_df.query("{}=='{}'".format(location_df.columns[i],modes[i])))/len(location_df[location_df.columns[i]]))*100,"%")
    else:
      print(location_df.columns[i],"   ",modes[i],"   ",(len(location_df.query("{}=={}".format(location_df.columns[i],modes[i])))/len(location_df[location_df.columns[i]]))*100,"%")
  
  duplicates = df[["longitude","latitude","date","time","day_of_week","location_easting_osgr", "location_northing_osgr"]].value_counts().reset_index(name="count")
  duplicates = duplicates[duplicates["count"]>1]
  duplicates

  from sklearn.neighbors import LocalOutlierFactor
  clf = LocalOutlierFactor(n_neighbors=20)
  subX = df[['number_of_vehicles','number_of_casualties']]
  X = subX.values
  y_pred = clf.fit_predict(X)

  plt.figure(figsize=(12,12))
  # plot the level sets of the decision function

  in_mask = [True if l == 1 else False for l in y_pred]
  out_mask = [True if l == -1 else False for l in y_pred]

  plt.title("Local Outlier Factor (LOF)")
  # inliers
  a = plt.scatter(X[in_mask, 0], X[in_mask, 1], c = 'blue',
                  edgecolor = 'k', s = 20)
  # outliers
  b = plt.scatter(X[out_mask, 0], X[out_mask, 1], c = 'red',
                  edgecolor = 'k', s = 20)
  plt.axis('tight')
  plt.xlabel('No. of vehicles');
  plt.ylabel('No. of casualties');
  plt.show()

  df_copy = df.copy()
  print("Rows before handling duplicate data: "+str(df_copy.shape[0]))
  subset_duplicates = ["longitude","latitude","date","time","day_of_week","location_easting_osgr", "location_northing_osgr"] 
  df_copy.drop_duplicates(subset=subset_duplicates, inplace=True)
  print("Rows after removing duplicates: "+str(df_copy.shape[0]))

  missing1 = df_copy.copy()
  print("Rows before removing missing data location "+str(missing1.shape[0]))
  missing1=missing1.dropna(subset=['location_easting_osgr','location_northing_osgr','longitude','latitude'])

  print("Rows after removing missing data location "+str(missing1.shape[0]))

  missing4 = missing1.copy()
  print("Number of missing values in the Road Type column: "+str(missing4['road_type'].isnull().sum()))
  my=missing4.groupby(["first_road_class"])["road_type"].agg(pd.Series.mode)
  missing4['road_type'] = missing4['road_type'].fillna(missing4['first_road_class'].map({'A':my[0],'A(M)': my[1], 'B':my[2],'C':my[3],'Motorway':my[4],'Unclassified':my[5]}))
  print("Number of missing values after imputing: "+str(missing4['road_type'].isnull().sum()))


  Junction = missing4.copy()
  print("Number of rows missing in the junction detail column: "+str(Junction[Junction['junction_detail']=='Data missing or out of range']['junction_detail'].count()))
  index_names = Junction[Junction['junction_detail'] == 'Data missing or out of range' ].index
  Junction.drop(index_names, inplace = True)
  print("Number of rows missing after dropping records: "+str(Junction[Junction['junction_detail']=='Data missing or out of range']['junction_detail'].count()))

  Junction_road = Junction.copy()
  print("Number of rows missing in the junction control column: "+str(Junction_road[Junction_road['junction_control']=='Data missing or out of range']['junction_control'].count()))
  Junction_road.loc[Junction_road.junction_detail == 'Not at junction or within 20 metres', 'junction_control'] = 'No junction'
  print("Number of rows missing in the junction control column after imputing: "+str(Junction_road[Junction_road['junction_control']=='Data missing or out of range']['junction_control'].count()))
  index_names = Junction_road[ Junction_road['junction_control'] == 'Data missing or out of range' ].index
  Junction_road.drop(index_names, inplace = True)
  print("Number of rows missing in the junction control column after dropping records: "+str(Junction_road[Junction_road['junction_control']=='Data missing or out of range']['junction_control'].count()))


  Junction_road1 = Junction_road.copy()
  print("Number of rows missing in the second road class column: "+str(Junction_road1[Junction_road1['second_road_class']=='-1']['second_road_class'].count()))
  Junction_road1.loc[Junction_road1.junction_detail == 'Not at junction or within 20 metres', 'second_road_class'] = 'No second road'
  print("Number of rows missing in the after imputing: "+str(Junction_road1[Junction_road1['second_road_class']=='-1']['second_road_class'].count()))
  index_names = Junction_road1[ Junction_road1['second_road_class'] == '-1' ].index
  Junction_road1.drop(index_names, inplace = True)
  print("Number of rows missing in the after dropping: "+str(Junction_road1[Junction_road1['second_road_class']=='-1']['second_road_class'].count()))

  Junction_road2 = Junction_road1.copy()
  print("Number of rows missing in the second road number column: "+str(Junction_road2['second_road_number'].isna().sum()))
  Junction_road2.loc[Junction_road2.junction_detail == 'Not at junction or within 20 metres', 'second_road_number'] = 'No second road'
  print("Number of rows missing after imputing: "+str(Junction_road2['second_road_number'].isna().sum()))
  index_names = Junction_road2[ Junction_road2['second_road_number'].isna()].index
  Junction_road2.drop(index_names, inplace = True)
  Junction_road2[Junction_road2['second_road_number'].isna()]
  print("Number of rows missing after dropping: "+str(Junction_road2['second_road_number'].isna().sum()))

  missing2 = Junction_road2.copy()
  print("Number of missing values: "+str(missing2['weather_conditions'].isnull().sum()))
  msss=missing2.groupby('did_police_officer_attend_scene_of_accident')['weather_conditions'].agg(lambda x:x.value_counts().index[0])
  missing2['weather_conditions'] = missing2['weather_conditions'].fillna(missing2['did_police_officer_attend_scene_of_accident'].map({'Yes':msss[3],'No - accident was reported using a self completion0form (self rep only)':msss[2],'No': msss[1],'Data missing or out of range':msss[0],np.nan:msss[0]}))
  print("Number of missing values after imputing: "+str(missing2['weather_conditions'].isna().sum()))
  df = missing2

  for d in df['date']:
    if isinstance(d,str):
        dt = datetime.strptime(d, '%d/%m/%Y')
        df['date'] = df['date'].replace([d],dt)
  weeks = []
  for d in df['date']:
    weeks.append(d.isocalendar()[1])
  df['week_number'] = weeks

  cleanup_nums = {"accident_severity":{"Slight": 0, "Serious": 1, "Fatal": 2},"urban_or_rural_area":{"Urban": 1, "Rural": 0}, "did_police_officer_attend_scene_of_accident":{"Yes": 1, "No": 0}, "trunk_road_flag":{"Non-trunk":0, "Trunk":1},"first_road_class":{"Unclassified": 0, "C": 1, "B": 2, "A": 3, "A(M)": 4,"Motorway":5 }, "second_road_class":{"Unclassified": 0, "C": 1, "B": 2, "A": 3, "A(M)": 4,"Motorway":5 } }
  encoded_df = df.replace(cleanup_nums)
  encoded_df

  accidents_df1=encoded_df.copy()

  one_hot3 = pd.get_dummies(accidents_df1['light_conditions'])
  # Drop column as it is now encoded
  accidents_df1 = accidents_df1.drop('light_conditions',axis = 1)
  # Join the encoded df
  accidents_df1 = accidents_df1.join(one_hot3)
  df = accidents_df1

  # Adding columns which adds more info
  is_weekend = []
  seasons = []
  for d in df['day_of_week']:
      if d in ['Friday', 'Saturday', 'Sunday']:
        is_weekend.append(True)
      else:
        is_weekend.append(False)
  #{0=spring, 1 = summer,  2= fall, 3 = winter}

  for d in df['date']:
      month = d.month
      if month in [3,4,5]:
          seasons.append(0)
      elif month in [6,7,8]:
          seasons.append(1)
      elif month in [9,10,11]:
          seasons.append(2)
      else:
          seasons.append(3)

  df['is_weekend'] = is_weekend
  df['season'] = seasons
  df.head()


  lookup = {"accident_severity":{"Slight": 0, "Serious": 1, "Fatal": 2},"urban_or_rural_area":{"Urban": 1, "Rural": 0}, "did_police_officer_attend_scene_of_accident":{"Yes": 1, "No": 0}, "trunk_road_flag":{"Non-trunk":0, "Trunk":1},"first_road_class":{"Unclassified": 0, "C": 1, "B": 2, "A": 3, "A(M)": 4,"Motorway":5 }, "second_road_class":{"Unclassified": 0, "C": 1, "B": 2, "A": 3, "A(M)": 4,"Motorway":5 }, "seasons":{"Spring":0, "Summer":1, "Fall":2, "Winter":3}}

  lookup_df = pd.DataFrame.from_dict(lookup)

  lookup_df.to_csv("./data/lookup.csv")

  df.to_csv('./data/output_dataset.csv')
