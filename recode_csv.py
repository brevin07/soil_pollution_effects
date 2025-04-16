import pandas as pd

# Load the CSV file
df = pd.read_csv('soil_pollution_diseases.csv')

# Recode Disease_Severity:
disease_severity_map = {
    "Mild": 0,
    "Moderate": 1,
    "Severe": 2
}
df['Disease_Severity'] = df['Disease_Severity'].map(disease_severity_map)

# Recode Pollutant_Type:
pollutant_type_map = {
    "Pesticides": 0,
    "Cadmium": 1,
    "Arsenic": 2,
    "Lead": 3,
    "mercury": 4,
    "Chromium": 5
}
df['Pollutant_Type'] = df['Pollutant_Type'].map(pollutant_type_map)

# Recode Farming_Practice:
farming_practice_map = {
    "Integrated": 0,
    "Permaculture": 1,
    "Organic": 2,
    "Conventional": 3
}
df['Farming_Practice'] = df['Farming_Practice'].map(farming_practice_map)

# For Nearby_Industry:
# First, fill in missing values (if any) so they are recoded as "NaN"
df['Nearby_Industry'] = df['Nearby_Industry'].fillna("NaN")
nearby_industry_map = {
    "NaN": 0,
    "Mining": 1,
    "Chemical": 2,
    "Textile": 3,
    "Agriculture": 4
}
df['Nearby_Industry'] = df['Nearby_Industry'].map(nearby_industry_map)

# Recode Age_Group_Affected
age_group_affected_map = {
    "Children" : 0,
    "Adults": 1,
    "Elderly" : 2
}
df['Age_Group_Affected'] = df['Age_Group_Affected'].map(age_group_affected_map)

# Optionally, save the recoded DataFrame to a new CSV file.
df.to_csv('soil_pollution_diseases_recoded.csv', index=False)

# Display the first few rows of the updated DataFrame
print(df.head().to_string())