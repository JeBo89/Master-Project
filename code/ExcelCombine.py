import pandas as pd

# Load data
Plie_list = []
Plie_data = []
Plie_xlsx = pd.ExcelFile('Plie_Keypoints.xlsx')
Plie_sheets = Plie_xlsx.sheet_names

for i in range(0,150):
    name = "Plie_"+str(i+1)
    if name in Plie_sheets:
        Plie_i = pd.read_excel(Plie_xlsx, name)
        Plie_list.append(name)
        Plie_data.append(Plie_i)
    else:
        continue

Jete_list = []
Jete_data = []
Jete_xlsx = pd.ExcelFile('Jete_Keypoints.xlsx')
Jete_sheets = Jete_xlsx.sheet_names

for i in range(0,150):
    name = "Jete_"+str(i+1)
    if name in Jete_sheets:
        Jete_i = pd.read_excel(Jete_xlsx, name)
        Jete_list.append(name)
        Jete_data.append(Jete_i)
    else:
        continue

Fondu_list = []
Fondu_data =[]
Fondu_xlsx = pd.ExcelFile('Fondu_Keypoints.xlsx')
Fondu_sheets = Fondu_xlsx.sheet_names

for i in range(0,150):
    name = "Fondu_"+str(i+1)
    if name in Fondu_sheets:
        Fondu_i = pd.read_excel(Fondu_xlsx, name)
        Fondu_list.append(name)
        Fondu_data.append(Fondu_i)
    else:
        continue

#Concentrate the list of dataframes:
Plie_df = pd.concat(Plie_data, keys=Plie_list)
Jete_df = pd.concat(Jete_data, keys=Jete_list)
Fondu_df = pd.concat(Fondu_data, keys=Fondu_list)

All_data = [Plie_df, Jete_df, Fondu_df]
All_data_df = pd.concat(All_data)

All_data_df.to_excel("AllData.xlsx") #--> Already done once

print("Data succesfully loaded")

# So now all the data is in stored per movement in a list (Plie_data,Jete_data and Fondu_data), which contains a list with the data of that movement.
# Each x,y,c data can be called upon using for example: Plie_data[0] for the first, etc....
