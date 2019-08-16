import pandas as pd

# Load data
Plie_sheets = []
Plie_xlsx = pd.ExcelFile('Plie_Keypoints_new.xlsx')
for sheet in Plie_xlsx.sheet_names:
    Plie_sheets.append(Plie_xlsx.parse(sheet))
    Plie = pd.concat(Plie_sheets)
#Plie['Label']='Plie'
print(Plie.shape)

Jete_sheets = []
Jete_xlsx = pd.ExcelFile('Jete_Keypoints_new.xlsx')
for sheet in Jete_xlsx.sheet_names:
    Jete_sheets.append(Jete_xlsx.parse(sheet))
    Jete = pd.concat(Jete_sheets)
#Jete['Label']='Jete'
print(Jete.shape)

Fondu_sheets = []
Fondu_xlsx = pd.ExcelFile('Fondu_Keypoints_new.xlsx')
for sheet in Fondu_xlsx.sheet_names:
    Fondu_sheets.append(Fondu_xlsx.parse(sheet))
    Fondu = pd.concat(Fondu_sheets)
#Fondu['Label']='Fondu'
print(Fondu.shape)

#Combine
All_keypoints = pd.concat([Plie, Jete, Fondu])
print(All_keypoints.shape)

All_keypoints.to_csv(r'C:\Users\jebo\Desktop\OpenPose Classi\keypoints_new.csv',index = None, header=True)
