import requests
import json
import pandas as pd

url = "https://www.ine.es/Censo2021/api"
path_data = "C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/"

### QUERY TO RETRIEVE EMPLOYMENT AT WORKPLACE
#query = {
#    "idioma": "EN",  # Use supported language
#    "metrica": ["SPERSONAS"],
#    "tabla": "viv.ppal",  
#    "variables": [
#        "ID_LUGAR_TRAB_N3"   
#        
#    ]
#}

### QUERY TO RETRIEVE ACTIVE PERSONS PER HOUSEHOLD
query = {
    "idioma": "EN",  # Use supported language
    "metrica": ["SHOGARES"],
    "tabla": "hog",  # Personal data table
    "variables": [
        "ID_RESIDENCIA_N4",  # Municipality level
        "ID_ACTI_HOG_2", "ID_ACTI_HOG_1"   # Province of workplace
        
    ]
}

response = requests.post(url, json=query, headers={
    "Content-Type": "application/json",
    "Accept": "application/json"
})

# Debug print
print("Status code:", response.status_code)
print("Response text (first 500 chars):", response.text[:500])

# If successful, parse content
if response.ok:
    content = response.json()
    datos = pd.DataFrame(content["data"])
else:
    print("Error: API call failed.")

# Extract the data
datos = pd.DataFrame(content["data"])
datos.to_csv(path_data + 'ppl_per_hh.csv')