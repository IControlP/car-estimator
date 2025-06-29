import streamlit as st
import pickle
import pandas as pd
import numpy as np
import math
from datetime import datetime
import plotly.graph_objects as go

# ─── Tier multipliers ──────────────────────────────────────────────────────────
tier_multipliers = {
    'Luxury':   1.2,
    'Midrange': 1.0,
    'Economy':  0.8
}

# ─── Load model and encoders ────────────────────────────────────────────────────
with open('car_maintenance_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)
with open('le_make.pkl', 'rb') as f:
    le_make = pickle.load(f)
with open('le_model.pkl', 'rb') as f:
    le_model = pickle.load(f)

# ─── MSRP and Edmunds ratings ──────────────────────────────────────────────────
msrp_data = {
    ('Acura','MDX'):51000,('Acura','RDX'):41000,('Acura','TLX'):39000,('Acura','ILX'):31000,('Acura','NSX'):157000,
    ('BMW','X1'):39000,('BMW','X3'):46000,('BMW','X5'):61000,('BMW','X7'):74000,
    ('Chevrolet','Malibu'):26000,('Chevrolet','Tahoe'):54000,('Chevrolet','Silverado'):47000,
    ('Ford','F-150'):35000,('Ford','Mustang'):31000,('Ford','Escape'):27000,
    ('Honda','Civic'):23000,('Honda','Accord'):28000,('Honda','CR-V'):29000,
    ('Hyundai','Elantra'):20000,('Hyundai','Sonata'):25000,('Hyundai','Tucson'):27000,
    ('Kia','Optima'):24000,('Kia','Soul'):21000,('Kia','Sportage'):26000,
    ('Lexus','ES'):42000,('Lexus','RX'):50000,('Lexus','NX'):42000,
    ('Mazda','3'):22000,('Mazda','6'):25000,('Mazda','CX-5'):28000,
    ('Mercedes-Benz','C-Class'):44000,('Mercedes-Benz','E-Class'):56000,('Mercedes-Benz','GLC'):49000,
    ('Mini','Cooper'):25000,('Mini','Countryman'):32000,
    ('Nissan','Altima'):25000,('Nissan','Sentra'):21000,('Nissan','Rogue'):28000,
    ('Porsche','911'):105000,('Porsche','Cayenne'):79000,
    ('Subaru','Impreza'):21000,('Subaru','Outback'):28000,('Subaru','Forester'):27000,
    ('Toyota','Camry'):27000,('Toyota','Corolla'):21000,('Toyota','RAV4'):29000,
    ('Volvo','XC60'):48000,('Volvo','XC90'):56000
}

# Vehicle ratings from accessible free sources (NHTSA, IIHS, Consumer Reports public data)
vehicle_ratings = {
    ('Acura','MDX'):{
        2020:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'5-star NHTSA safety, IIHS Top Safety Pick, reliable luxury SUV'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Maintained excellent safety ratings, refreshed design'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Enhanced to Top Safety Pick+, improved LED headlights'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Continued safety excellence, technology updates'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Maintained top-tier safety performance'}
    },
    ('BMW','X1'):{
        2020:{'nhtsa':5,'iihs':'Good','desc':'5-star NHTSA rating, solid IIHS performance'},
        2021:{'nhtsa':5,'iihs':'Good','desc':'Consistent safety performance, tech updates'},
        2022:{'nhtsa':5,'iihs':'Good','desc':'Final year of generation, maintained quality'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'All-new generation, improved to TSP status'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Refined safety systems, enhanced features'}
    },
    ('BMW','X3'):{
        2020:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Excellent safety across all categories'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Enhanced headlight performance for TSP+'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Maintained premium safety standards'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Continued excellence in luxury SUV safety'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Industry-leading safety performance'}
    },
    ('Honda','Civic'):{
        2020:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'5-star NHTSA, TSP award, excellent reliability'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Enhanced Honda Sensing safety suite'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'All-new generation achieves TSP+ status'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Maintained top safety performance'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Continued compact car safety leadership'}
    },
    ('Honda','Accord'):{
        2020:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'5-star NHTSA rating, TSP award'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Consistent midsize sedan safety excellence'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Enhanced to TSP+ with improved lighting'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Refreshed design maintains TSP+ status'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Industry-leading midsize sedan safety'}
    },
    ('Honda','CR-V'):{
        2020:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Popular compact SUV with excellent safety'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Maintained TSP status with special editions'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Updated styling retains safety excellence'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'All-new generation achieves TSP+ award'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Enhanced safety tech and efficiency'}
    },
    ('Toyota','Camry'):{
        2020:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'5-star NHTSA rating, reliable midsize sedan'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Enhanced Toyota Safety Sense 2.0'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Improved to TSP+ with lighting upgrades'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Maintained TSP+ with consistent quality'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Enhanced safety tech and reliability'}
    },
    ('Toyota','Corolla'):{
        2020:{'nhtsa':5,'iihs':'Good','desc':'5-star NHTSA rating, excellent reliability'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Enhanced to TSP with improved safety tech'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Maintained TSP status with updates'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Continued compact car safety leadership'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Refined Toyota Safety Sense features'}
    },
    ('Toyota','RAV4'):{
        2020:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Popular compact SUV with standard AWD'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Prime PHEV variant maintains TSP status'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Enhanced to TSP+ with improved lighting'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Continued compact SUV safety leadership'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Refreshed design maintains TSP+ award'}
    },
    ('Ford','F-150'):{
        2020:{'nhtsa':5,'iihs':'Good','desc':'5-star NHTSA rating, America\'s best-selling truck'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'All-new generation with enhanced safety tech'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Lightning EV variant maintains safety excellence'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Continued truck safety leadership'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Advanced driver assistance systems'}
    },
    ('Chevrolet','Malibu'):{
        2020:{'nhtsa':5,'iihs':'Good','desc':'5-star overall NHTSA rating, good IIHS scores'},
        2021:{'nhtsa':5,'iihs':'Good','desc':'Maintained safety performance, updated features'},
        2022:{'nhtsa':5,'iihs':'Good','desc':'Consistent midsize sedan safety'},
        2023:{'nhtsa':5,'iihs':'Good','desc':'Final production year, maintained standards'},
        2024:{'nhtsa':None,'iihs':None,'desc':'Model discontinued'}
    },
    ('Nissan','Altima'):{
        2020:{'nhtsa':5,'iihs':'Good','desc':'5-star NHTSA rating with ProPILOT Assist'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Enhanced to TSP with improved safety tech'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Refreshed design maintains TSP status'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Enhanced Safety Shield 360 technology'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick','desc':'Continued midsize sedan safety leadership'}
    },
    ('Subaru','Outback'):{
        2020:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'All-new generation with TSP+ award'},
        2021:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Enhanced EyeSight driver assistance'},
        2022:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Wilderness variant maintains TSP+ status'},
        2023:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Continued adventure vehicle safety excellence'},
        2024:{'nhtsa':5,'iihs':'Top Safety Pick+','desc':'Refined EyeSight and safety systems'}
    }
}

# ─── Fuel requirements by make/model ────────────────────────────────────────────
fuel_requirements = {
    'Acura': {'MDX': 'premium', 'RDX': 'premium', 'TLX': 'premium', 'ILX': 'regular', 'NSX': 'premium'},
    'BMW': {'X1': 'premium', 'X3': 'premium', 'X5': 'premium', 'X7': 'premium', 
            'M3': 'premium', 'M4': 'premium', 'M5': 'premium', 'M8': 'premium',
            '328i': 'premium', '530i': 'premium', '750i': 'premium', 'i4': 'electric', 'iX': 'electric'},
    'Chevrolet': {'Malibu': 'regular', 'Tahoe': 'regular', 'Silverado': 'regular', 'Equinox': 'regular'},
    'Ford': {'F-150': 'regular', 'Mustang': 'premium', 'Escape': 'regular'},
    'Honda': {'Civic': 'regular', 'Accord': 'regular', 'CR-V': 'regular'},
    'Hyundai': {'Elantra': 'regular', 'Sonata': 'regular', 'Tucson': 'regular'},
    'Kia': {'Optima': 'regular', 'Soul': 'regular', 'Sportage': 'regular'},
    'Lexus': {'ES': 'premium', 'RX': 'premium', 'NX': 'premium'},
    'Mazda': {'3': 'regular', '6': 'regular', 'CX-5': 'regular'},
    'Mercedes-Benz': {'C-Class': 'premium', 'E-Class': 'premium', 'GLC': 'premium'},
    'Mini': {'Cooper': 'premium', 'Countryman': 'premium'},
    'Nissan': {'Altima': 'regular', 'Sentra': 'regular', 'Rogue': 'regular'},
    'Porsche': {'911': 'premium', 'Cayenne': 'premium'},
    'Subaru': {'Impreza': 'regular', 'Outback': 'regular', 'Forester': 'regular'},
    'Toyota': {'Camry': 'regular', 'Corolla': 'regular', 'RAV4': 'regular'},
    'Volvo': {'XC60': 'premium', 'XC90': 'premium'},
    'Tesla': {'Model 3': 'electric', 'Model S': 'electric', 'Model X': 'electric', 'Model Y': 'electric'}
}

# ─── State fuel prices (regular/premium) ───────────────────────────────────────
state_fuel_prices = {
    'Alabama': {'regular': 3.20, 'premium': 3.90}, 'Alaska': {'regular': 3.80, 'premium': 4.50},
    'Arizona': {'regular': 3.45, 'premium': 4.15}, 'Arkansas': {'regular': 3.15, 'premium': 3.85},
    'California': {'regular': 4.85, 'premium': 5.55}, 'Colorado': {'regular': 3.40, 'premium': 4.10},
    'Connecticut': {'regular': 3.65, 'premium': 4.35}, 'Delaware': {'regular': 3.35, 'premium': 4.05},
    'Florida': {'regular': 3.30, 'premium': 4.00}, 'Georgia': {'regular': 3.25, 'premium': 3.95},
    'Hawaii': {'regular': 4.20, 'premium': 4.90}, 'Idaho': {'regular': 3.55, 'premium': 4.25},
    'Illinois': {'regular': 3.75, 'premium': 4.45}, 'Indiana': {'regular': 3.35, 'premium': 4.05},
    'Iowa': {'regular': 3.25, 'premium': 3.95}, 'Kansas': {'regular': 3.20, 'premium': 3.90},
    'Kentucky': {'regular': 3.30, 'premium': 4.00}, 'Louisiana': {'regular': 3.10, 'premium': 3.80},
    'Maine': {'regular': 3.50, 'premium': 4.20}, 'Maryland': {'regular': 3.55, 'premium': 4.25},
    'Massachusetts': {'regular': 3.70, 'premium': 4.40}, 'Michigan': {'regular': 3.45, 'premium': 4.15},
    'Minnesota': {'regular': 3.40, 'premium': 4.10}, 'Mississippi': {'regular': 3.05, 'premium': 3.75},
    'Missouri': {'regular': 3.15, 'premium': 3.85}, 'Montana': {'regular': 3.60, 'premium': 4.30},
    'Nebraska': {'regular': 3.25, 'premium': 3.95}, 'Nevada': {'regular': 3.85, 'premium': 4.55},
    'New Hampshire': {'regular': 3.45, 'premium': 4.15}, 'New Jersey': {'regular': 3.50, 'premium': 4.20},
    'New Mexico': {'regular': 3.35, 'premium': 4.05}, 'New York': {'regular': 3.75, 'premium': 4.45},
    'North Carolina': {'regular': 3.25, 'premium': 3.95}, 'North Dakota': {'regular': 3.30, 'premium': 4.00},
    'Ohio': {'regular': 3.40, 'premium': 4.10}, 'Oklahoma': {'regular': 3.10, 'premium': 3.80},
    'Oregon': {'regular': 3.85, 'premium': 4.55}, 'Pennsylvania': {'regular': 3.65, 'premium': 4.35},
    'Rhode Island': {'regular': 3.60, 'premium': 4.30}, 'South Carolina': {'regular': 3.20, 'premium': 3.90},
    'South Dakota': {'regular': 3.25, 'premium': 3.95}, 'Tennessee': {'regular': 3.15, 'premium': 3.85},
    'Texas': {'regular': 3.00, 'premium': 3.70}, 'Utah': {'regular': 3.55, 'premium': 4.25},
    'Vermont': {'regular': 3.65, 'premium': 4.35}, 'Virginia': {'regular': 3.35, 'premium': 4.05},
    'Washington': {'regular': 4.10, 'premium': 4.80}, 'West Virginia': {'regular': 3.40, 'premium': 4.10},
    'Wisconsin': {'regular': 3.35, 'premium': 4.05}, 'Wyoming': {'regular': 3.45, 'premium': 4.15}
}

# ─── Lookup tables ─────────────────────────────────────────────────────────────
car_makes_and_models = {
    'Acura': ['MDX', 'RDX', 'TLX', 'ILX', 'NSX'],
    'Alfa Romeo': ['Giulia', 'Stelvio', '4C'],
    'Audi': ['A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Q3', 'Q5', 'Q7', 'Q8', 'Q5 Sportback', 'S4', 'RS7'],
    'BMW': ['X1', 'X3', 'X5', 'X7', 'M3', 'M4', 'M5', 'M8', '328i', '530i', '750i', 'i4', 'iX'],
    'Buick': ['Enclave', 'Encore', 'LaCrosse', 'Regal', 'Envision'],
    'Cadillac': ['Escalade', 'XT5', 'CTS', 'ATS', 'CT4', 'CT5', 'XT4', 'SRX'],
    'Chevrolet': ['Equinox', 'Malibu', 'Silverado 1500', 'Traverse', 'Tahoe', 'Impala', 'Colorado', 'Suburban', 'Spark', 'Silverado'],
    'Chrysler': ['Pacifica', 'Voyager', '300'],
    'Dodge': ['Charger', 'Durango', 'Ram 1500', 'Challenger', 'Grand Caravan'],
    'Fiat': ['500', '500X', '124 Spider'],
    'Ford': ['F-150', 'Mustang', 'Explorer', 'Escape', 'Bronco', 'Edge', 'Ranger', 'Fusion', 'Expedition'],
    'GMC': ['Sierra 1500', 'Yukon', 'Canyon', 'Acadia', 'Terrain'],
    'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot', 'Odyssey', 'Ridgeline', 'Insight'],
    'Hyundai': ['Elantra', 'Sonata', 'Tucson', 'Santa Fe', 'Kona', 'Palisade', 'Veloster'],
    'Infiniti': ['Q50', 'Q60', 'QX60', 'QX80', 'QX50'],
    'Jaguar': ['F-PACE', 'XE', 'XJ', 'F-TYPE'],
    'Jeep': ['Grand Cherokee', 'Wrangler', 'Cherokee', 'Compass', 'Renegade'],
    'Kia': ['Sorento', 'Optima', 'Stinger', 'Sportage', 'Telluride', 'Seltos', 'Niro', 'Soul'],
    'Lexus': ['RX', 'ES', 'NX', 'GX', 'LX', 'IS', 'LS', 'UX'],
    'Lincoln': ['Navigator', 'MKX', 'Corsair', 'MKZ'],
    'Mazda': ['CX-5', 'Mazda 3', 'Mazda 6', 'MX-5 Miata', 'CX-9'],
    'McLaren': ['720S', '570S', 'GT', '765LT'],
    'Mercedes-Benz': ['C-Class', 'E-Class', 'GLC', 'S-Class', 'GLS', 'A-Class', 'CLA', 'G-Class', 'AMG GT'],
    'Mini': ['Cooper', 'Countryman', 'Clubman'],
    'Mitsubishi': ['Outlander', 'Eclipse Cross', 'Mirage'],
    'Nissan': ['Altima', 'Maxima', 'Sentra', 'Murano', 'Rogue', 'Titan', '370Z'],
    'Porsche': ['911', 'Macan', 'Cayenne', 'Panamera', 'Taycan'],
    'Ram': ['1500', '2500', '3500', 'ProMaster'],
    'Subaru': ['Outback', 'Forester', 'Impreza', 'Crosstrek', 'Legacy'],
    'Tesla': ['Model 3', 'Model S', 'Model X', 'Model Y'],
    'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Tacoma', 'Tundra', 'Land Cruiser', 'Sequoia'],
    'Volkswagen': ['Passat', 'Jetta', 'Tiguan', 'Atlas', 'Golf'],
    'Volvo': ['XC90', 'XC60', 'S60', 'V90', 'XC40']
}

# ─── Vehicle Lifespan Data ─────────────────────────────────────────────────────
# Expected vehicle lifespan in years based on make/model reliability data
vehicle_lifespan = {
    'Toyota': {'default': 20, 'Camry': 22, 'Corolla': 25, 'RAV4': 20, 'Highlander': 18, 'Tacoma': 25, 'Tundra': 22, 'Land Cruiser': 30, 'Sequoia': 20},
    'Honda': {'default': 20, 'Civic': 22, 'Accord': 20, 'CR-V': 18, 'Pilot': 17, 'Odyssey': 16, 'Ridgeline': 20, 'Insight': 18},
    'Lexus': {'default': 18, 'ES': 20, 'RX': 18, 'NX': 16, 'GX': 22, 'LX': 25, 'IS': 18, 'LS': 20, 'UX': 16},
    'Acura': {'default': 16, 'MDX': 16, 'RDX': 15, 'TLX': 16, 'ILX': 16, 'NSX': 12},
    'Subaru': {'default': 18, 'Outback': 20, 'Forester': 18, 'Impreza': 18, 'Crosstrek': 17, 'Legacy': 18},
    'Mazda': {'default': 16, '3': 16, '6': 16, 'CX-5': 15, 'Mazda3': 16, 'Mazda6': 16, 'MX-5 Miata': 18, 'CX-9': 15},
    'Nissan': {'default': 15, 'Altima': 15, 'Sentra': 14, 'Rogue': 14, 'Maxima': 16, 'Murano': 15, 'Titan': 18, '370Z': 14},
    'Hyundai': {'default': 14, 'Elantra': 14, 'Sonata': 15, 'Tucson': 13, 'Santa Fe': 14, 'Kona': 12, 'Palisade': 14, 'Veloster': 12},
    'Kia': {'default': 14, 'Optima': 14, 'Soul': 13, 'Sportage': 13, 'Sorento': 14, 'Stinger': 12, 'Telluride': 14, 'Seltos': 12, 'Niro': 14},
    'Ford': {'default': 14, 'F-150': 18, 'Mustang': 15, 'Escape': 13, 'Explorer': 14, 'Bronco': 16, 'Edge': 13, 'Ranger': 16, 'Fusion': 14, 'Expedition': 16},
    'Chevrolet': {'default': 13, 'Malibu': 13, 'Tahoe': 16, 'Silverado': 18, 'Equinox': 12, 'Silverado 1500': 18, 'Traverse': 13, 'Impala': 14, 'Colorado': 16, 'Suburban': 18, 'Spark': 10},
    'GMC': {'default': 14, 'Sierra 1500': 18, 'Yukon': 16, 'Canyon': 16, 'Acadia': 13, 'Terrain': 12},
    'BMW': {'default': 12, 'X1': 12, 'X3': 13, 'X5': 14, 'X7': 12, 'M3': 10, 'M4': 10, 'M5': 10, 'M8': 10, '328i': 12, '530i': 13, '750i': 12, 'i4': 15, 'iX': 15},
    'Mercedes-Benz': {'default': 12, 'C-Class': 12, 'E-Class': 13, 'GLC': 12, 'S-Class': 14, 'GLS': 13, 'A-Class': 11, 'CLA': 11, 'G-Class': 20, 'AMG GT': 10},
    'Audi': {'default': 12, 'A3': 11, 'A4': 12, 'A5': 12, 'A6': 13, 'A7': 12, 'A8': 13, 'Q3': 11, 'Q5': 12, 'Q7': 13, 'Q8': 12, 'Q5 Sportback': 12, 'S4': 10, 'RS7': 10},
    'Volvo': {'default': 14, 'XC90': 15, 'XC60': 14, 'S60': 13, 'V90': 14, 'XC40': 12},
    'Mini': {'default': 11, 'Cooper': 11, 'Countryman': 12, 'Clubman': 11},
    'Porsche': {'default': 12, '911': 15, 'Cayenne': 12, 'Macan': 11, 'Panamera': 12, 'Taycan': 15},
    'Jaguar': {'default': 10, 'F-PACE': 10, 'XE': 9, 'XJ': 11, 'F-TYPE': 10},
    'Tesla': {'default': 15, 'Model 3': 15, 'Model S': 16, 'Model X': 14, 'Model Y': 15},
    'Jeep': {'default': 12, 'Grand Cherokee': 13, 'Wrangler': 15, 'Cherokee': 12, 'Compass': 11, 'Renegade': 10},
    'Ram': {'default': 16, '1500': 16, '2500': 18, '3500': 20, 'ProMaster': 14},
    'Cadillac': {'default': 11, 'Escalade': 13, 'XT5': 11, 'CTS': 12, 'ATS': 10, 'CT4': 11, 'CT5': 12, 'XT4': 10, 'SRX': 11},
    'Lincoln': {'default': 12, 'Navigator': 13, 'MKX': 12, 'Corsair': 11, 'MKZ': 12},
    'Infiniti': {'default': 12, 'Q50': 12, 'Q60': 11, 'QX60': 13, 'QX80': 14, 'QX50': 11},
    'Buick': {'default': 13, 'Enclave': 13, 'Encore': 12, 'LaCrosse': 14, 'Regal': 13, 'Envision': 12},
    'Chrysler': {'default': 11, 'Pacifica': 11, 'Voyager': 10, '300': 12},
    'Dodge': {'default': 11, 'Charger': 12, 'Durango': 12, 'Ram 1500': 16, 'Challenger': 13, 'Grand Caravan': 10},
    'Mitsubishi': {'default': 12, 'Outlander': 12, 'Eclipse Cross': 11, 'Mirage': 10},
    'Volkswagen': {'default': 11, 'Passat': 11, 'Jetta': 12, 'Tiguan': 11, 'Atlas': 11, 'Golf': 12},
    'Fiat': {'default': 8, '500': 8, '500X': 9, '124 Spider': 10},
    'McLaren': {'default': 8, '720S': 8, '570S': 8, 'GT': 9, '765LT': 7},
    'Alfa Romeo': {'default': 9, 'Giulia': 9, 'Stelvio': 10, '4C': 8}
}

def get_vehicle_lifespan(make, model):
    """Get expected vehicle lifespan in years"""
    make_data = vehicle_lifespan.get(make, {'default': 12})
    return make_data.get(model, make_data.get('default', 12))

def get_max_forecast_years(make, model, current_year, model_year):
    """Calculate maximum realistic forecast years - now allows up to 30 years"""
    vehicle_age = current_year - model_year
    # Allow forecasting up to 30 years total, regardless of expected lifespan
    max_possible_age = 30
    remaining_years = max(1, max_possible_age - vehicle_age)
    
    return min(remaining_years, 30)

# State electricity rates ($/kWh) for residential use
state_electricity_rates = {
    'Alabama': 0.1421, 'Alaska': 0.2298, 'Arizona': 0.1378, 'Arkansas': 0.1140,
    'California': 0.2855, 'Colorado': 0.1378, 'Connecticut': 0.2406, 'Delaware': 0.1355,
    'Florida': 0.1302, 'Georgia': 0.1268, 'Hawaii': 0.4018, 'Idaho': 0.1089,
    'Illinois': 0.1371, 'Indiana': 0.1465, 'Iowa': 0.1421, 'Kansas': 0.1418,
    'Kentucky': 0.1198, 'Louisiana': 0.1089, 'Maine': 0.1640, 'Maryland': 0.1421,
    'Massachusetts': 0.2298, 'Michigan': 0.1640, 'Minnesota': 0.1421, 'Mississippi': 0.1235,
    'Missouri': 0.1198, 'Montana': 0.1140, 'Nebraska': 0.1089, 'Nevada': 0.1235,
    'New Hampshire': 0.1888, 'New Jersey': 0.1640, 'New Mexico': 0.1355, 'New York': 0.2051,
    'North Carolina': 0.1198, 'North Dakota': 0.1089, 'Ohio': 0.1355, 'Oklahoma': 0.1235,
    'Oregon': 0.1140, 'Pennsylvania': 0.1465, 'Rhode Island': 0.2051, 'South Carolina': 0.1355,
    'South Dakota': 0.1198, 'Tennessee': 0.1198, 'Texas': 0.1235, 'Utah': 0.1140,
    'Vermont': 0.1888, 'Virginia': 0.1235, 'Washington': 0.1037, 'West Virginia': 0.1198,
    'Wisconsin': 0.1421, 'Wyoming': 0.1140
}

# EV charging efficiency and consumption data
ev_charging_data = {
    'Tesla': {
        'Model 3': {
            'battery_kwh': 75, 'efficiency_miles_per_kwh': 4.0, 'range_miles': 300,
            'home_charging_loss': 0.10, 'public_charging_loss': 0.15
        },
        'Model S': {
            'battery_kwh': 100, 'efficiency_miles_per_kwh': 3.4, 'range_miles': 340,
            'home_charging_loss': 0.10, 'public_charging_loss': 0.15
        },
        'Model X': {
            'battery_kwh': 100, 'efficiency_miles_per_kwh': 3.0, 'range_miles': 300,
            'home_charging_loss': 0.10, 'public_charging_loss': 0.15
        },
        'Model Y': {
            'battery_kwh': 75, 'efficiency_miles_per_kwh': 3.7, 'range_miles': 280,
            'home_charging_loss': 0.10, 'public_charging_loss': 0.15
        }
    },
    'BMW': {
        'i4': {
            'battery_kwh': 84, 'efficiency_miles_per_kwh': 3.5, 'range_miles': 294,
            'home_charging_loss': 0.12, 'public_charging_loss': 0.18
        },
        'iX': {
            'battery_kwh': 106, 'efficiency_miles_per_kwh': 2.8, 'range_miles': 297,
            'home_charging_loss': 0.12, 'public_charging_loss': 0.18
        }
    },
    'Porsche': {
        'Taycan': {
            'battery_kwh': 93, 'efficiency_miles_per_kwh': 2.4, 'range_miles': 223,
            'home_charging_loss': 0.12, 'public_charging_loss': 0.18
        }
    }
}

# Time-of-use electricity rates for EV optimization ($/kWh)
time_of_use_rates = {
    'California': {'peak': 0.52, 'off_peak': 0.16, 'ev_rate': 0.13},
    'Texas': {'peak': 0.18, 'off_peak': 0.08, 'ev_rate': 0.10},
    'Florida': {'peak': 0.16, 'off_peak': 0.09, 'ev_rate': 0.11},
    'New York': {'peak': 0.28, 'off_peak': 0.12, 'ev_rate': 0.14},
    'Illinois': {'peak': 0.19, 'off_peak': 0.08, 'ev_rate': 0.10},
    'Arizona': {'peak': 0.20, 'off_peak': 0.08, 'ev_rate': 0.09},
    'Washington': {'peak': 0.14, 'off_peak': 0.06, 'ev_rate': 0.07},
    'Massachusetts': {'peak': 0.32, 'off_peak': 0.14, 'ev_rate': 0.16},
    'Colorado': {'peak': 0.18, 'off_peak': 0.08, 'ev_rate': 0.10},
    'Georgia': {'peak': 0.16, 'off_peak': 0.08, 'ev_rate': 0.09}
}

# ─── Vehicle MPG/Efficiency Data ───────────────────────────────────────────────
average_mpg = {
    'Acura': {
        'MDX': 22, 'RDX': 23, 'TLX': 24, 'ILX': 25, 'NSX': 21
    },
    'BMW': {
        'X1': 28, 'X3': 25, 'X5': 22, 'X7': 21,
        'M3': 20, 'M4': 19, 'M5': 18, 'M8': 17,
        '328i': 30, '530i': 29, '750i': 22,
        'i4': 103, 'iX': 80
    },
    'Chevrolet': {
        'Malibu': 29, 'Tahoe': 20, 'Silverado': 23,
        'Equinox': 26, 'Silverado 1500': 20, 'Traverse': 22,
        'Impala': 19, 'Colorado': 20, 'Suburban': 19, 'Spark': 30
    },
    'Ford': {
        'F-150': 20, 'Mustang': 24, 'Escape': 27,
        'Explorer': 24, 'Bronco': 21, 'Edge': 24,
        'Ranger': 23, 'Fusion': 25, 'Expedition': 17
    },
    'Honda': {
        'Civic': 32, 'Accord': 30, 'CR-V': 29,
        'Pilot': 22, 'Odyssey': 28, 'Ridgeline': 21, 'Insight': 52
    },
    'Hyundai': {
        'Elantra': 33, 'Sonata': 31, 'Tucson': 26,
        'Santa Fe': 25, 'Kona': 28, 'Palisade': 22, 'Veloster': 28
    },
    'Kia': {
        'Optima': 27, 'Soul': 28, 'Sportage': 26,
        'Sorento': 25, 'Stinger': 22, 'Telluride': 21, 'Seltos': 29, 'Niro': 50
    },
    'Lexus': {
        'ES': 26, 'RX': 23, 'NX': 25,
        'GX': 16, 'LX': 16, 'IS': 26, 'LS': 23, 'UX': 29
    },
    'Mazda': {
        '3': 28, '6': 26, 'CX-5': 25,
        'Mazda3': 28, 'Mazda6': 26, 'MX-5 Miata': 26, 'CX-9': 22
    },
    'Mercedes-Benz': {
        'C-Class': 25, 'E-Class': 24, 'GLC': 24,
        'S-Class': 21, 'GLS': 19, 'A-Class': 30, 'CLA': 27, 'G-Class': 13, 'AMG GT': 18
    },
    'Mini': {
        'Cooper': 29, 'Countryman': 27, 'Clubman': 27
    },
    'Nissan': {
        'Altima': 32, 'Sentra': 29, 'Rogue': 28,
        'Maxima': 23, 'Murano': 25, 'Titan': 18, '370Z': 19
    },
    'Porsche': {
        '911': 22, 'Cayenne': 22,
        'Macan': 21, 'Panamera': 22, 'Taycan': 72
    },
    'Subaru': {
        'Impreza': 31, 'Outback': 29, 'Forester': 28,
        'Crosstrek': 28, 'Legacy': 29
    },
    'Toyota': {
        'Camry': 32, 'Corolla': 33, 'RAV4': 28,
        'Highlander': 24, 'Tacoma': 20, 'Tundra': 17, 'Land Cruiser': 14, 'Sequoia': 17
    },
    'Volvo': {
        'XC60': 24, 'XC90': 21, 'S60': 26, 'V90': 25, 'XC40': 28
    },
    'Tesla': {'Model 3': 120, 'Model S': 102, 'Model X': 90, 'Model Y': 112},
}

# ─── Maintenance Schedules & Costs ─────────────────────────────────────────────

# Regular ICE maintenance schedule
maintenance_schedule = {
    'Oil Change':5000,'Tire Rotation':7500,'Cabin Air Filter Replacement':15000,
    'Brake Inspection':20000,'Coolant Flush':30000,'Battery Check':30000,
    'Brake Pad Replacement':40000,'Transmission Fluid Replacement':60000,
    'Spark Plug Replacement':80000,'Alternator Inspection':90000,
    'Timing Belt Replacement':100000,'Wheel Bearings Check':110000,
    'AC System Service':120000,'Head Gasket Inspection':130000,
    'Major Overhaul Recommended':150000
}

# Electric vehicle maintenance schedule
ev_maintenance_schedule = {
    'Tire Rotation':7500,'Cabin Air Filter Replacement':15000,
    'Brake Inspection':25000,'Battery Coolant Check':30000,'12V Battery Check':30000,
    'Brake Pad Replacement':60000,'Drive Unit Service':60000,
    'HVAC Filter Replacement':24000,'Wheel Bearings Check':100000,
    'AC System Service':120000,'Battery Health Check':100000
}

# Enhanced maintenance costs with labor/parts breakdown
maintenance_costs = {
    # Regular maintenance
    'Oil Change': {'labor': 25, 'parts': 35, 'total': 60},
    'Tire Rotation': {'labor': 25, 'parts': 0, 'total': 25},
    'Cabin Air Filter Replacement': {'labor': 20, 'parts': 25, 'total': 45},
    'Brake Inspection': {'labor': 40, 'parts': 0, 'total': 40},
    'Coolant Flush': {'labor': 60, 'parts': 40, 'total': 100},
    '12V Battery Check': {'labor': 20, 'parts': 0, 'total': 20},
    'Battery Check': {'labor': 30, 'parts': 0, 'total': 30},
    'Brake Pad Replacement': {'labor': 80, 'parts': 120, 'total': 200},
    'Transmission Fluid Replacement': {'labor': 75, 'parts': 85, 'total': 160},
    'Spark Plug Replacement': {'labor': 60, 'parts': 80, 'total': 140},
    'Alternator Inspection': {'labor': 60, 'parts': 0, 'total': 60},
    'Timing Belt Replacement': {'labor': 450, 'parts': 350, 'total': 800},
    'Wheel Bearings Check': {'labor': 100, 'parts': 0, 'total': 100},
    'AC System Service': {'labor': 90, 'parts': 60, 'total': 150},
    'Head Gasket Inspection': {'labor': 200, 'parts': 0, 'total': 200},
    'Major Overhaul Recommended': {'labor': 2000, 'parts': 1500, 'total': 3500},
    
    # EV-specific maintenance
    'Battery Coolant Check': {'labor': 40, 'parts': 20, 'total': 60},
    'Drive Unit Service': {'labor': 120, 'parts': 80, 'total': 200},
    'HVAC Filter Replacement': {'labor': 30, 'parts': 40, 'total': 70},
    'Battery Health Check': {'labor': 80, 'parts': 0, 'total': 80},
    
    # Age-related maintenance (appears in older vehicles)
    'Transmission Service': {'labor': 150, 'parts': 150, 'total': 300},
    'Suspension Check': {'labor': 100, 'parts': 50, 'total': 150},
    'Engine Mount Replacement': {'labor': 400, 'parts': 400, 'total': 800},
    'CV Joint Replacement': {'labor': 300, 'parts': 250, 'total': 550},
    'Power Steering Service': {'labor': 80, 'parts': 70, 'total': 150},
    'Radiator Replacement': {'labor': 200, 'parts': 300, 'total': 500},
    'Catalytic Converter Replacement': {'labor': 150, 'parts': 800, 'total': 950},
    
    # Extreme aging maintenance (beyond expected lifespan)
    'Battery Pack Degradation Service': {'labor': 500, 'parts': 1500, 'total': 2000},
    'Drive Unit Overhaul': {'labor': 800, 'parts': 1200, 'total': 2000},
    'Engine Overhaul': {'labor': 2000, 'parts': 3000, 'total': 5000},
    'Transmission Rebuild': {'labor': 1500, 'parts': 2000, 'total': 3500}
}

# ─── State Cost Multipliers ────────────────────────────────────────────────────
state_cost_multipliers = {
    'Alabama':1.00,'Alaska':1.10,'Arizona':1.05,'Arkansas':0.95,
    'California':1.25,'Colorado':1.10,'Connecticut':1.20,'Delaware':1.05,
    'Florida':1.00,'Georgia':1.00,'Hawaii':1.30,'Idaho':0.95,
    'Illinois':1.10,'Indiana':0.95,'Iowa':0.90,'Kansas':0.95,
    'Kentucky':0.95,'Louisiana':1.00,'Maine':1.05,'Maryland':1.20,
    'Massachusetts':1.25,'Michigan':1.00,'Minnesota':1.00,'Mississippi':0.90,
    'Missouri':0.95,'Montana':0.95,'Nebraska':0.95,'Nevada':1.05,
    'New Hampshire':1.10,'New Jersey':1.25,'New Mexico':0.95,'New York':1.30,
    'North Carolina':1.00,'North Dakota':0.90,'Ohio':0.95,'Oklahoma':0.95,
    'Oregon':1.10,'Pennsylvania':1.10,'Rhode Island':1.15,'South Carolina':0.95,
    'South Dakota':0.90,'Tennessee':0.95,'Texas':1.00,'Utah':1.00,
    'Vermont':1.10,'Virginia':1.10,'Washington':1.15,'West Virginia':0.90,
    'Wisconsin':1.00,'Wyoming':0.90
}

# ─── Electric Vehicle Functions ────────────────────────────────────────────────

def is_electric_vehicle(make, model):
    """Check if the vehicle is electric"""
    return fuel_requirements.get(make, {}).get(model) == 'electric'

def calculate_ev_electricity_cost(make, model, avg_mpy, state, charging_preference='mixed', custom_rates=None):
    """
    Calculate accurate electricity costs for EVs based on vehicle efficiency and state rates
    
    Args:
        make, model: Vehicle identification
        avg_mpy: Annual mileage
        state: State for electricity rates
        charging_preference: 'home', 'public', or 'mixed'
        custom_rates: Optional dict with custom rates {'residential': rate, 'ev_rate': rate, 'public': rate, 'has_ev_rate': bool}
    """
    if not is_electric_vehicle(make, model):
        return 0
    
    # Get vehicle-specific data
    ev_data = ev_charging_data.get(make, {}).get(model)
    if not ev_data:
        # Fallback for EVs not in our database
        ev_data = {
            'efficiency_miles_per_kwh': 3.0,
            'home_charging_loss': 0.12,
            'public_charging_loss': 0.18
        }
    
    # Calculate kWh needed per year (accounting for charging losses)
    base_kwh_needed = avg_mpy / ev_data['efficiency_miles_per_kwh']
    
    # Use custom rates if provided, otherwise use default state rates
    if custom_rates:
        base_rate = custom_rates['residential']
        ev_rate = custom_rates['ev_rate'] if custom_rates['has_ev_rate'] else custom_rates['residential']
        public_rate = custom_rates['public']
    else:
        # Get default electricity rates
        base_rate = state_electricity_rates.get(state, 0.15)
        tou_rates = time_of_use_rates.get(state, {'ev_rate': base_rate})
        ev_rate = tou_rates.get('ev_rate', base_rate)
        public_rate = 0.35  # Default public charging rate
    
    if charging_preference == 'home':
        # Mostly home charging with EV rate or regular rate
        rate = ev_rate
        kwh_with_losses = base_kwh_needed * (1 + ev_data['home_charging_loss'])
        annual_cost = kwh_with_losses * rate
        
    elif charging_preference == 'public':
        # Mostly public charging (more expensive)
        kwh_with_losses = base_kwh_needed * (1 + ev_data['public_charging_loss'])
        annual_cost = kwh_with_losses * public_rate
        
    else:  # mixed
        # 70% home, 30% public charging
        home_kwh = base_kwh_needed * 0.7 * (1 + ev_data['home_charging_loss'])
        public_kwh = base_kwh_needed * 0.3 * (1 + ev_data['public_charging_loss'])
        
        annual_cost = (home_kwh * ev_rate) + (public_kwh * public_rate)
    
    return annual_cost

def get_ev_charging_info(make, model, state):
    """Get detailed EV charging information for display"""
    if not is_electric_vehicle(make, model):
        return None
    
    ev_data = ev_charging_data.get(make, {}).get(model)
    base_rate = state_electricity_rates.get(state, 0.15)
    tou_rates = time_of_use_rates.get(state, {})
    
    info = {
        'base_rate': base_rate,
        'has_ev_rate': 'ev_rate' in tou_rates,
        'ev_rate': tou_rates.get('ev_rate', base_rate),
        'efficiency': ev_data.get('efficiency_miles_per_kwh', 3.0) if ev_data else 3.0,
        'battery_size': ev_data.get('battery_kwh', 75) if ev_data else 75,
        'range': ev_data.get('range_miles', 250) if ev_data else 250
    }
    
    return info

def get_scheduled_activities(start_mileage, end_mileage, is_ev=False, driving_style='normal', terrain='flat'):
    """Get maintenance activities with adjustments for driving conditions"""
    schedule = ev_maintenance_schedule if is_ev else maintenance_schedule
    activities = []
    
    # Adjust intervals based on driving style and terrain
    style_multiplier = {'gentle': 1.2, 'normal': 1.0, 'aggressive': 0.8}[driving_style]
    terrain_multiplier = {'flat': 1.0, 'hilly': 0.85}[terrain]
    
    adjustment_factor = style_multiplier * terrain_multiplier
    
    for name, base_interval in schedule.items():
        adjusted_interval = int(base_interval * adjustment_factor)
        next_due = ((start_mileage // adjusted_interval) + 1) * adjusted_interval
        if start_mileage < next_due <= end_mileage:
            activities.append(name)
    
    return activities

def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 0.0

def depreciation_fraction(t,V0,Vmin=2000.0,a=0.182):
    return (1 - Vmin/V0)*(1 - safe_exp(-a*t))

def residual_value(t,V0,Vmin=2000.0,a=0.182):
    return V0*(1 - depreciation_fraction(t,V0,Vmin,a))

def estimate_vehicle_value(msrp,model_year,current_year,Vmin=2000.0,a=0.182):
    elapsed=max(0,current_year-model_year)
    elapsed=min(elapsed,50)
    return round(residual_value(elapsed,msrp,Vmin,a),2)

def get_car_tier(make):
    lux={'BMW','Mercedes-Benz','Audi','Lexus','Jaguar','Porsche','Volvo','Mini','McLaren','Acura','Cadillac','Lincoln','Infiniti'}
    eco={'Toyota','Honda','Ford','Hyundai','Kia','Chevrolet','Subaru','Mazda','Nissan','Mitsubishi','Chrysler','Dodge'}
    if make in lux: return 'Luxury'
    if make in eco: return 'Economy'
    return 'Midrange'

def get_fuel_price(state, make, model, custom_price=None):
    """Get appropriate fuel price based on vehicle requirements and state, with optional custom price"""
    fuel_type = fuel_requirements.get(make, {}).get(model, 'regular')
    if fuel_type == 'electric':
        return 0  # No fuel cost for electric vehicles
    
    # Use custom price if provided, otherwise use state default
    if custom_price is not None:
        return custom_price
    
    state_prices = state_fuel_prices.get(state, {'regular': 3.50, 'premium': 4.20})
    return state_prices.get(fuel_type, state_prices['regular'])

def predict_5_years_cost(make, model, model_year, current_mileage, avg_mpy,
                        mpg, purchase_price, state,
                        years, loan_amount, irate, lt_years,
                        user_age, start_age, msrp, driving_style, terrain):
    
    is_ev = is_electric_vehicle(make, model)
    fuel_price = get_fuel_price(state, make, model)
    
    enc_make = le_make.transform([make])[0]
    enc_model = le_model.transform([model])[0]
    
    # Enhanced tier multiplier for parts costs
    tier = get_car_tier(make)
    parts_multiplier = {'Luxury': 1.8, 'Midrange': 1.2, 'Economy': 1.0}[tier]
    labor_multiplier = tier_multipliers[tier]
    state_mult = state_cost_multipliers[state]
    
    # Get vehicle lifespan for aging calculations
    expected_lifespan = get_vehicle_lifespan(make, model)
    current_vehicle_age = datetime.now().year - model_year
    
    r = irate / 100 if irate > 0 else 0
    loan_pay = (loan_amount * r / (1 - (1 + r) ** -lt_years)) if r > 0 and loan_amount > 0 else (loan_amount / lt_years if loan_amount > 0 else 0)
    
    prev_val = purchase_price
    Y, M, F, L, D, T, V, Acts, Lines, Insurance = [], [], [], [], [], [], [], [], [], []
    
    for i in range(1, years + 1):
        yr = model_year + i
        fm = current_mileage + avg_mpy * i
        vehicle_age_in_year = current_vehicle_age + i
        
        # Calculate aging multiplier for maintenance costs - enhanced for extreme aging
        aging_multiplier = 1.0
        if vehicle_age_in_year > expected_lifespan * 0.6:  # After 60% of expected lifespan
            # Costs increase exponentially as vehicle ages
            excess_years = vehicle_age_in_year - (expected_lifespan * 0.6)
            aging_multiplier = 1.0 + (excess_years * 0.15)  # 15% increase per year after 60% of lifespan
            
        if vehicle_age_in_year > expected_lifespan * 0.8:  # After 80% of expected lifespan
            excess_years = vehicle_age_in_year - (expected_lifespan * 0.8)
            aging_multiplier += (excess_years * 0.25)  # Additional 25% increase per year after 80%
            
        if vehicle_age_in_year > expected_lifespan:  # Beyond expected lifespan
            excess_years = vehicle_age_in_year - expected_lifespan
            aging_multiplier += (excess_years * 0.50)  # Additional 50% increase per year beyond lifespan
            
        if vehicle_age_in_year > expected_lifespan + 5:  # Well beyond expected lifespan
            excess_years = vehicle_age_in_year - (expected_lifespan + 5)
            aging_multiplier += (excess_years * 0.75)  # Additional 75% increase for very old vehicles
            
        aging_multiplier = min(aging_multiplier, 8.0)  # Cap at 8x normal costs for extreme cases
        
        # Base maintenance prediction
        if not is_ev:
            df = pd.DataFrame([{
                'Make_Encoded': enc_make, 'Model_Encoded': enc_model,
                'Year': model_year, 'Mileage': fm, 'Avg_Miles_Per_Year': avg_mpy
            }])
            base = trained_model.predict(df)[0] * aging_multiplier
        else:
            # Simplified base cost for EVs (no oil changes, etc.)
            base = 200 * (1 + (fm / 100000) * 0.5) * aging_multiplier
        
        # Get scheduled activities
        acts = get_scheduled_activities(
            current_mileage + avg_mpy * (i - 1), fm, is_ev, driving_style, terrain
        )
        
        # Add age-related maintenance items - enhanced for extreme aging
        if vehicle_age_in_year >= expected_lifespan * 0.7 and not is_ev:
            # Add common high-mileage issues
            if vehicle_age_in_year >= expected_lifespan * 0.7 and 'Transmission Service' not in acts:
                acts.append('Transmission Service')
            if vehicle_age_in_year >= expected_lifespan * 0.8 and 'Suspension Check' not in acts:
                acts.append('Suspension Check')
                
        if vehicle_age_in_year > expected_lifespan and not is_ev:
            # Beyond expected lifespan - major component issues
            if 'Engine Mount Replacement' not in acts:
                acts.append('Engine Mount Replacement')
            if vehicle_age_in_year > expected_lifespan + 2 and 'CV Joint Replacement' not in acts:
                acts.append('CV Joint Replacement')
            if vehicle_age_in_year > expected_lifespan + 3 and 'Power Steering Service' not in acts:
                acts.append('Power Steering Service')
                
        if vehicle_age_in_year > expected_lifespan + 5 and not is_ev:
            # Very old vehicles - extreme maintenance
            if 'Radiator Replacement' not in acts and vehicle_age_in_year % 3 == 0:
                acts.append('Radiator Replacement')
            if 'Catalytic Converter Replacement' not in acts and vehicle_age_in_year % 4 == 0:
                acts.append('Catalytic Converter Replacement')
                
        # EV-specific extreme aging issues
        if is_ev and vehicle_age_in_year > expected_lifespan:
            if 'Battery Pack Degradation Service' not in acts:
                acts.append('Battery Pack Degradation Service')
            if vehicle_age_in_year > expected_lifespan + 3 and 'Drive Unit Overhaul' not in acts:
                acts.append('Drive Unit Overhaul')
        
        # Calculate activity costs with enhanced breakdown
        act_cost = 0
        for activity in acts:
            if activity in maintenance_costs:
                labor_cost = maintenance_costs[activity]['labor'] * labor_multiplier * state_mult * aging_multiplier
                parts_cost = maintenance_costs[activity]['parts'] * parts_multiplier * state_mult * aging_multiplier
                act_cost += labor_cost + parts_cost
            else:
                # Handle new age-related activities
                if activity == 'Transmission Service':
                    act_cost += 300 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'Suspension Check':
                    act_cost += 150 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'Engine Mount Replacement':
                    act_cost += 800 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'CV Joint Replacement':
                    act_cost += 550 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'Power Steering Service':
                    act_cost += 150 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'Radiator Replacement':
                    act_cost += 500 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'Catalytic Converter Replacement':
                    act_cost += 950 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'Battery Pack Degradation Service':
                    act_cost += 2000 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'Drive Unit Overhaul':
                    act_cost += 2000 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'Engine Overhaul':
                    act_cost += 5000 * labor_multiplier * state_mult * aging_multiplier
                elif activity == 'Transmission Rebuild':
                    act_cost += 3500 * labor_multiplier * state_mult * aging_multiplier
        
        maint = base + act_cost
        
        # Fuel/electricity cost
        if is_ev:
            # Use custom rates if available from the UI
            if 'custom_rates_dict' in locals() and custom_rates_dict.get('use_custom', False):
                custom_rates = {
                    'residential': custom_rates_dict['residential'],
                    'ev_rate': custom_rates_dict['ev_rate'],
                    'public': custom_rates_dict['public'],
                    'has_ev_rate': custom_rates_dict['has_ev_rate']
                }
            else:
                custom_rates = None
            
            fuel = calculate_ev_electricity_cost(make, model, avg_mpy, state, 'mixed', custom_rates)
            
            # EVs may become less efficient as they age (battery degradation)
            if vehicle_age_in_year > 8:  # After 8 years, some efficiency loss
                battery_degradation = 1 + ((vehicle_age_in_year - 8) * 0.02)  # 2% per year after 8 years
                fuel *= battery_degradation
        else:
            # Use custom fuel price if available
            custom_price = custom_fuel_price if 'custom_fuel_price' in locals() else None
            fuel_price = get_fuel_price(state, make, model, custom_price)
            
            # ICE vehicles may become less efficient as they age
            efficiency_loss = 1.0
            if vehicle_age_in_year > 10:
                efficiency_loss = 1 + ((vehicle_age_in_year - 10) * 0.01)  # 1% loss per year after 10 years
            
            fuel = (avg_mpy / mpg) * fuel_price * efficiency_loss
        
        reg = 150
        total = maint + fuel + reg + loan_pay
        
        # Enhanced depreciation for older vehicles
        cur_val = estimate_vehicle_value(purchase_price, model_year, yr)
        
        # Accelerated depreciation near end of life
        if vehicle_age_in_year > expected_lifespan * 0.9:
            # Value drops more rapidly as vehicle approaches end of useful life
            end_of_life_factor = (vehicle_age_in_year - expected_lifespan * 0.9) / (expected_lifespan * 0.1)
            end_of_life_factor = min(end_of_life_factor, 1.0)
            # Reduce value by up to 50% more than normal depreciation
            additional_depreciation = cur_val * 0.3 * end_of_life_factor
            cur_val = max(cur_val - additional_depreciation, 500)  # Minimum scrap value
        
        depr = round(prev_val - cur_val, 2)
        prev_val = cur_val
        
        # Insurance calculation with age adjustments
        this_year_age = user_age + i - 1
        years_driving = this_year_age - start_age
        premium = 1200  # base
        if this_year_age < 25:
            premium += 700
        if this_year_age >= 25 and years_driving >= 10:
            premium -= 200
        if msrp and msrp > 50000:
            premium += 400
        if avg_mpy > 15000:
            premium += 200
        elif avg_mpy < 7000:
            premium -= 100
            
        # Older vehicles may have lower insurance costs due to lower value
        if vehicle_age_in_year > 10:
            age_discount = min(0.3, (vehicle_age_in_year - 10) * 0.03)  # Up to 30% discount
            premium *= (1 - age_discount)
            
        premium *= state_cost_multipliers.get(state, 1.0)
        premium = max(700, round(premium))
        
        Y.append(f"Year {i}")
        M.append(round(maint, 2))
        F.append(round(fuel, 2))
        L.append(round(loan_pay, 2))
        D.append(depr)
        T.append(round(total, 2))
        V.append(cur_val)
        Acts.append(', '.join(acts) or 'None')
        Lines.append(f"Year {i}: Maintenance ${round(maint):,}, {'Electricity' if is_ev else 'Fuel'} ${round(fuel):,}, Loan ${round(loan_pay):,}, Depreciation ${depr:,}\\n")
        Insurance.append(premium)
    
    df_out = pd.DataFrame({
        'Year': Y,
        'Maintenance Cost': M,
        'Fuel/Electricity Cost': F,
        'Loan Payment': L,
        'Depreciation Cost': D,
        'Total Cost': T,
        'Car Value': V,
        'Activities': Acts,
        'Insurance Premium': Insurance
    })
    
    inter = next((y for y, m, v in zip(Y, M, V) if m > v), None)
    return ''.join(Lines), df_out, inter

# ─── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("🚗 Car Ownership Cost Forecast")
st.markdown("""
**Protect your financial future with smart vehicle decisions.**

Whether you're shopping for a new or used vehicle, this tool goes beyond the sticker price and monthly payments to reveal the **true cost of ownership**. We're not here to tell you if you're getting the best deal on the car itself – we're here to help you understand the **complete financial commitment** of owning that vehicle.

**Our mission is financial protection and empowerment:**
- **Prevent Financial Overextension**: Understand the full cost before you commit
- **Enable Socioeconomic Mobility**: Make informed decisions that protect your financial growth
- **Avoid Hidden Financial Traps**: See costs that dealers and sellers don't discuss
- **Empower Non-Car People**: No automotive expertise needed – just honest financial analysis

**What makes this different:**
- **Total Financial Reality**: See maintenance, fuel, insurance, depreciation, and financing costs
- **Budget Protection**: Determine if this specific vehicle could harm your financial stability
- **Smart Comparisons**: Compare electric vs gas vehicles with real-world cost breakdowns
- **Financial Education**: Learn what car ownership really costs over time

**Perfect for:**
- 🛡️ **Financial Protection**: Avoid purchases that could damage your economic future
- 🎓 **First-Time Buyers**: Learn what car ownership actually costs beyond payments
- ⚡ **EV Consideration**: Understand if electric vehicles make financial sense for you
- 📊 **Budget Planning**: Ensure transportation costs don't derail other financial goals
- 👥 **Non-Car People**: Get clear, honest analysis without automotive jargon

**Remember:** A car that seems affordable today can become a financial burden tomorrow. We help you see the complete picture before you sign – because your financial future matters more than any vehicle.
""")

# Vehicle Selection
st.header("🔧 Vehicle Information")
col1, col2 = st.columns(2)

with col1:
    make = st.selectbox(
        "Make",
        sorted(car_makes_and_models.keys()),
        key="make"
    )

with col2:
    model = st.selectbox(
        "Model",
        car_makes_and_models[make],
        key="model"
    )

model_year = st.selectbox(
    "Model Year",
    list(range(2000, datetime.now().year+1))[::-1],
    key="model_year"
)

# Enhanced MSRP & safety rating display
msrp = msrp_data.get((make, model))
rating_data = vehicle_ratings.get((make, model), {}).get(model_year)

col1, col2 = st.columns(2)
with col1:
    if msrp:
        st.markdown(f"**MSRP (New):** ${msrp:,.0f}")
    else:
        st.markdown("**MSRP:** Not available")

with col2:
    if rating_data:
        nhtsa_rating = rating_data.get('nhtsa')
        iihs_rating = rating_data.get('iihs')
        description = rating_data['desc']
        
        if nhtsa_rating:
            st.markdown(f"**NHTSA Safety:** {nhtsa_rating}/5 ⭐")
        if iihs_rating and iihs_rating != 'Good':
            st.markdown(f"**IIHS Award:** {iihs_rating} 🏆")
        elif iihs_rating:
            st.markdown(f"**IIHS Rating:** {iihs_rating}")
            
        st.caption(f"*{description}*")
    else:
        st.markdown("**Safety Ratings:** Not available")

st.caption("⭐ Safety ratings from NHTSA (National Highway Traffic Safety Administration) and IIHS (Insurance Institute for Highway Safety) - free public data sources")

# Vehicle Details
st.header("📊 Vehicle Details")
col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input(
        "Current Mileage",
        0, 300000, 1000,
        key="mileage"
    )

with col2:
    your_price = st.number_input(
        "Your Purchase Price ($)",
        0, 2000000, 20000,
        key="your_price"
    )

# Check if it's an electric vehicle
is_ev = is_electric_vehicle(make, model)

# Location and Forecast Period - Move this up before EV-specific sections
st.header("📍 Location & Forecast")

state = st.selectbox(
    "State",
    sorted(state_cost_multipliers.keys()),
    key="state"
)

# Fuel/electricity pricing section
if is_ev:
    base_rate = state_electricity_rates.get(state, 0.15)
    st.caption(f"Default residential electricity rate in {state}: ${base_rate:.3f}/kWh")
    tou_info = time_of_use_rates.get(state)
    if tou_info and 'ev_rate' in tou_info:
        st.caption(f"EV time-of-use rate available: ${tou_info['ev_rate']:.3f}/kWh")
else:
    # Gas price customization for non-EV vehicles
    fuel_type = fuel_requirements.get(make, {}).get(model, 'regular')
    state_prices = state_fuel_prices.get(state, {'regular': 3.50, 'premium': 4.20})
    default_fuel_price = state_prices.get(fuel_type, state_prices['regular'])
    
    st.subheader("⛽ Fuel Pricing")
    fuel_emoji = "⛽" if fuel_type == "regular" else "🏎️"
    st.markdown(f"{fuel_emoji} This vehicle requires **{fuel_type}** gasoline")
    
    # User input for fuel price with state default pre-filled
    custom_fuel_price = st.number_input(
        f"{fuel_type.title()} Gas Price ($/gallon)",
        min_value=1.00,
        max_value=10.00,
        value=default_fuel_price,
        step=0.01,
        format="%.2f",
        key="custom_fuel_price",
        help=f"Default {fuel_type} gas price for {state} is ${default_fuel_price:.2f}/gallon. Adjust based on your local prices."
    )
    
    # Show comparison to state average
    if abs(custom_fuel_price - default_fuel_price) > 0.05:
        price_diff = custom_fuel_price - default_fuel_price
        if price_diff > 0:
            st.caption(f"💰 ${price_diff:.2f}/gallon above {state} average")
        else:
            st.caption(f"💰 ${abs(price_diff):.2f}/gallon below {state} average")
    else:
        st.caption(f"📍 Close to {state} average price")

# Enhanced EV information display
if is_ev:
    st.info("🔋 **Electric Vehicle Detected** - Specialized calculations for electricity costs and EV maintenance will be applied.")
    
    ev_info = get_ev_charging_info(make, model, state)
    if ev_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Battery Size", f"{ev_info['battery_size']} kWh")
        with col2:
            st.metric("Efficiency", f"{ev_info['efficiency']:.1f} mi/kWh")
        with col3:
            st.metric("EPA Range", f"{ev_info['range']} miles")
        
        # Electricity rate configuration
        st.subheader("⚡ Electricity Rate Configuration")
        st.markdown("*Customize your electricity rates for more accurate cost calculations*")
        
        # Option to use default rates or customize
        use_custom_rates = st.checkbox(
            "Customize Electricity Rates",
            value=False,
            help="Check this to override default state rates with your actual rates"
        )
        
        if use_custom_rates:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏠 Home Charging Rates**")
                
                # Residential rate input
                custom_residential_rate = st.number_input(
                    "Residential Rate ($/kWh)",
                    min_value=0.01,
                    max_value=1.00,
                    value=ev_info['base_rate'],
                    step=0.001,
                    format="%.3f",
                    key="custom_residential_rate",
                    help="Your home electricity rate per kWh"
                )
                
                # EV time-of-use rate input
                has_ev_tou = st.checkbox(
                    "I have a special EV time-of-use rate",
                    value=ev_info['has_ev_rate'],
                    key="has_custom_ev_rate"
                )
                
                if has_ev_tou:
                    custom_ev_rate = st.number_input(
                        "EV Time-of-Use Rate ($/kWh)",
                        min_value=0.01,
                        max_value=1.00,
                        value=ev_info['ev_rate'],
                        step=0.001,
                        format="%.3f",
                        key="custom_ev_rate",
                        help="Special EV charging rate (usually for off-peak hours)"
                    )
                else:
                    custom_ev_rate = custom_residential_rate
            
            with col2:
                st.markdown("**🚗 Public Charging Rates**")
                
                # Public charging rate input
                default_public_rate = 0.35  # Default public charging rate
                custom_public_rate = st.number_input(
                    "Public Charging Rate ($/kWh)",
                    min_value=0.10,
                    max_value=1.00,
                    value=default_public_rate,
                    step=0.01,
                    format="%.2f",
                    key="custom_public_rate",
                    help="Average rate at public charging stations"
                )
                
                st.caption("💡 **Tips for finding your rates:**")
                st.caption("• Check your electricity bill for kWh rates")
                st.caption("• Look for EV time-of-use plans from your utility")
                st.caption("• Public rates vary by network (Tesla, ChargePoint, etc.)")
            
            # Store custom rates for later use in calculations
            custom_rates_dict = {
                'residential': custom_residential_rate,
                'ev_rate': custom_ev_rate,
                'public': custom_public_rate,
                'has_ev_rate': has_ev_tou,
                'use_custom': True
            }
            
        else:
            # Display default rates
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"🏠 **Default Residential Rate:** ${ev_info['base_rate']:.3f}/kWh")
                if ev_info['has_ev_rate']:
                    st.write(f"⚡ **Default EV Time-of-Use Rate:** ${ev_info['ev_rate']:.3f}/kWh")
                else:
                    st.write("⚡ No special EV rate available in your state")
            
            with col2:
                st.write(f"🚗 **Default Public Charging Rate:** $0.35/kWh")
                st.caption("Based on national average for public charging")
            
            # Set to use default rates
            custom_rates_dict = {'use_custom': False}
        
        # Charging preference
        charging_pref = st.selectbox(
            "Primary Charging Method",
            ["mixed", "home", "public"],
            format_func=lambda x: {
                "mixed": "Mixed (70% home, 30% public)",
                "home": "Mostly Home Charging",
                "public": "Mostly Public Charging"
            }[x],
            key="charging_pref",
            help="This affects your electricity costs significantly"
        )
    else:
        charging_pref = "mixed"
        custom_rates_dict = {'use_custom': False}
else:
    charging_pref = "mixed"
    custom_rates_dict = {'use_custom': False}

default_mpg = average_mpg.get(make, {}).get(model, 25)
if is_ev:
    mpg_label = "Miles Per kWh Equivalent"
    mpg_help = f"EPA avg: {default_mpg} MPGe"
else:
    mpg_label = "Miles Per Gallon"
    mpg_help = f"EPA avg: {default_mpg} MPG"

mpg = st.number_input(
    mpg_label,
    10, 150, value=default_mpg,
    help=mpg_help,
    key="mpg"
)
st.caption(f"EPA Average for {make} {model}: {default_mpg} {'MPGe' if is_ev else 'MPG'}")

# Financing
st.header("💰 Financing Information")
col1, col2, col3 = st.columns(3)

with col1:
    loan_amount = st.number_input(
        "Loan Amount ($)",
        0.0, 2000000.0, 0.0,
        key="loan_amount",
        help="Enter 0 if paying cash"
    )

with col2:
    irate = st.number_input(
        "Interest Rate (%)",
        0.0, 25.0, 5.0,
        key="irate"
    )

with col3:
    lt_years = st.number_input(
        "Loan Term (years)",
        1, 8, 3,
        key="lt_years"
    )

# Personal Information
st.header("👤 Personal Information")
col1, col2 = st.columns(2)

with col1:
    gross = st.number_input(
        "Gross Annual Income ($)",
        0.0, 1000000.0, 60000.0,
        key="gross"
    )

with col2:
    avg_mpy = st.number_input(
        "Average Miles Per Year",
        0, 100000, 10000,
        key="avg_mpy",
        help="The average American drives about 10,000-12,000 miles per year"
    )
st.caption("ℹ️ Average annual mileage in the US is approximately 10,000-12,000 miles")

# Driving conditions
st.header("🛣️ Driving Conditions")
st.markdown("*These factors affect maintenance schedules and costs*")

col1, col2 = st.columns(2)
with col1:
    driving_style = st.selectbox(
        "Driving Style",
        ["gentle", "normal", "aggressive"],
        index=1,
        key="driving_style",
        help="Gentle: Easy acceleration/braking, Normal: Average driving, Aggressive: Hard acceleration/braking"
    )

with col2:
    terrain = st.selectbox(
        "Terrain",
        ["flat", "hilly"],
        key="terrain",
        help="Hilly terrain puts more strain on the vehicle"
    )

# Insurance Information
st.header("🛡️ Insurance Information")
st.markdown("*We ask for your age to estimate insurance premiums, which vary significantly by age and driving experience*")

col1, col2 = st.columns(2)
with col1:
    user_age = st.number_input(
        "Your Age",
        min_value=16, max_value=100, value=30,
        key="user_age"
    )

with col2:
    start_age = st.number_input(
        "Age When You Started Driving",
        min_value=14, max_value=user_age, value=16,
        key="start_age"
    )

# Final Configuration
st.header("🔮 Forecast Configuration")

# Calculate maximum forecast years based on vehicle lifespan
max_years = get_max_forecast_years(make, model, datetime.now().year, model_year)
expected_lifespan = get_vehicle_lifespan(make, model)
vehicle_age = datetime.now().year - model_year

# Display vehicle lifespan information
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Vehicle Age", f"{vehicle_age} years")
with col2:
    st.metric("Expected Lifespan", f"{expected_lifespan} years")
with col3:
    st.metric("Max Forecast", f"{max_years} years")

# Provide context about the lifespan estimate
st.info(f"""
**📊 Vehicle Lifespan Information:**
- **{make} {model}** typically lasts **{expected_lifespan} years** based on reliability data
- Your vehicle is **{vehicle_age} years old**
- You can forecast up to **{max_years} years** (maximum 30 years total vehicle age)
- Costs beyond expected lifespan include significantly higher maintenance and repair risks
""")

# Show reliability context
reliability_context = ""
if expected_lifespan >= 20:
    reliability_context = "🏆 **Excellent Reliability** - This vehicle is known for exceptional longevity"
elif expected_lifespan >= 16:
    reliability_context = "✅ **Above Average Reliability** - This vehicle typically outlasts most others"
elif expected_lifespan >= 13:
    reliability_context = "📊 **Average Reliability** - Typical lifespan for this vehicle class"
elif expected_lifespan >= 10:
    reliability_context = "⚠️ **Below Average Reliability** - Higher maintenance costs expected as it ages"
else:
    reliability_context = "🚨 **Lower Reliability** - Consider replacement planning due to shorter expected lifespan"

st.markdown(reliability_context)

years = st.slider(
    "Years to Forecast",
    1, max_years, min(5, max_years),
    key="years",
    help=f"Select how many years to forecast. You can forecast up to {max_years} years, even beyond the vehicle's expected lifespan of {expected_lifespan} years."
)

# Enhanced warnings based on forecast duration
final_vehicle_age = vehicle_age + years
years_beyond_lifespan = max(0, final_vehicle_age - expected_lifespan)

if years_beyond_lifespan > 0:
    st.error(f"""
    🚨 **Beyond Expected Lifespan Warning**: 
    You're forecasting {years_beyond_lifespan} year(s) beyond this vehicle's expected {expected_lifespan}-year lifespan.
    
    **Expect significantly higher costs:**
    - Major component failures (engine, transmission, suspension)
    - Frequent breakdowns and emergency repairs
    - Difficulty finding parts for very old vehicles
    - Potential safety and reliability issues
    - Diminishing returns on repair investments
    """)
elif final_vehicle_age > expected_lifespan * 0.8:
    st.warning(f"""
    ⚠️ **Approaching End-of-Life Warning**: 
    Your vehicle will be {final_vehicle_age} years old, which is {((final_vehicle_age/expected_lifespan)*100):.0f}% of its expected lifespan.
    Expect significantly higher maintenance costs and potential major repairs.
    """)

# Show cost impact preview
if years_beyond_lifespan > 0:
    st.markdown(f"""
    **💰 Extended Ownership Cost Impact:**
    - **Years 1-{expected_lifespan - vehicle_age}**: Normal aging cost increases
    - **Years {expected_lifespan - vehicle_age + 1}-{years}**: Extreme cost escalation (up to 4x normal maintenance)
    - **Major repair risk**: Budget for potential engine/transmission replacement
    """)

# Add lifetime cost estimate with extended forecasting
if st.checkbox("📈 Show Full 30-Year Ownership Projection", help="See complete ownership costs if you kept this vehicle for 30 years total"):
    max_possible_years = min(30 - vehicle_age, 30)
    if max_possible_years > 0:
        lifetime_summary, lifetime_df, _ = predict_5_years_cost(
            make, model, model_year, mileage, avg_mpy,
            mpg, your_price, state, max_possible_years, loan_amount, irate, lt_years,
            user_age, start_age, msrp, driving_style, terrain
        )
        
        if lifetime_summary:
            lifetime_total = lifetime_df['Total Cost'].sum()
            lifetime_avg = lifetime_total / max_possible_years
            
            # Split costs by normal vs extended periods
            normal_years = min(max_possible_years, expected_lifespan - vehicle_age)
            extended_years = max(0, max_possible_years - normal_years)
            
            if extended_years > 0:
                normal_cost = lifetime_df['Total Cost'].iloc[:normal_years].sum() if normal_years > 0 else 0
                extended_cost = lifetime_df['Total Cost'].iloc[normal_years:].sum() if extended_years > 0 else 0
                
                st.warning(f"""
                **🔮 Full 30-Year Projection ({max_possible_years} years remaining):**
                - **Total Cost:** ${lifetime_total:,.0f}
                - **Normal Years (1-{normal_years}):** ${normal_cost:,.0f} (${normal_cost/normal_years:,.0f}/year)
                - **Extended Years ({normal_years+1}-{max_possible_years}):** ${extended_cost:,.0f} (${extended_cost/extended_years:,.0f}/year)
                - **Cost Per Mile:** ${(lifetime_total / (avg_mpy * max_possible_years)):.3f}
                """)
            else:
                st.success(f"""
                **🔮 Lifetime Ownership Estimate ({max_possible_years} years):**
                - **Total Cost:** ${lifetime_total:,.0f}
                - **Average Annual Cost:** ${lifetime_avg:,.0f}
                - **Cost Per Mile:** ${(lifetime_total / (avg_mpy * max_possible_years)):.3f}
                """)
            
            # Show major milestones including beyond-lifespan issues
            major_milestones = []
            for i, row in lifetime_df.iterrows():
                year_num = i + 1
                activities = row['Activities']
                vehicle_age_at_year = vehicle_age + year_num
                
                if ('Major Overhaul' in activities or 'Timing Belt' in activities or 
                    'Transmission' in activities or vehicle_age_at_year > expected_lifespan):
                    milestone_warning = ""
                    if vehicle_age_at_year > expected_lifespan:
                        milestone_warning = " ⚠️ BEYOND EXPECTED LIFESPAN"
                    major_milestones.append(f"Year {year_num}: {activities} (${row['Maintenance Cost']:,.0f}){milestone_warning}")
            
            if major_milestones:
                st.warning("**🔧 Major Maintenance & Repair Milestones:**")
                for milestone in major_milestones[:5]:  # Show first 5 major items
                    st.markdown(f"• {milestone}")
                if len(major_milestones) > 5:
                    st.caption(f"...and {len(major_milestones) - 5} more major maintenance items")

if st.button("🔮 Predict Ownership Costs", type="primary"):
    # Run the prediction
    summary, chart_df, intersection = predict_5_years_cost(
        make, model, model_year, mileage, avg_mpy,
        mpg, your_price, state, years, loan_amount, irate, lt_years,
        user_age, start_age, msrp, driving_style, terrain
    )
    
    if not summary:
        st.error("Prediction failed—check inputs.")
        st.stop()

    # Refresh mileage & scheduled activities
    chart_df['Mileage'] = [
        mileage + avg_mpy * i
        for i in range(1, years + 1)
    ]
    
    # Show forecast table
    st.header("📊 Detailed Forecast Results")
    
    # Configure pandas display options for better text wrapping
    pd.set_option('display.max_colwidth', None)
    
    # Style the dataframe for better readability
    styled_df = chart_df.style.set_properties(**{
        'text-align': 'left',
        'white-space': 'pre-wrap',
        'word-wrap': 'break-word',
        'max-width': '300px'
    }, subset=['Activities'])
    
    styled_df = styled_df.set_properties(**{
        'text-align': 'right'
    }, subset=['Maintenance Cost', 'Fuel/Electricity Cost', 'Loan Payment', 'Depreciation Cost', 'Total Cost', 'Car Value', 'Insurance Premium'])
    
    styled_df = styled_df.format({
        'Maintenance Cost': '${:,.0f}',
        'Fuel/Electricity Cost': '${:,.0f}',
        'Loan Payment': '${:,.0f}',
        'Depreciation Cost': '${:,.0f}',
        'Total Cost': '${:,.0f}',
        'Car Value': '${:,.0f}',
        'Insurance Premium': '${:,.0f}',
        'Mileage': '{:,.0f}'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    fuel_label = "electricity" if is_ev else "fuel"
    st.caption(f"Includes maintenance, {fuel_label}, registration, loan payments, and insurance premiums.")
    
    # Add expandable section for detailed maintenance breakdown
    with st.expander("🔧 Detailed Maintenance Activities by Year"):
        current_vehicle_age = datetime.now().year - model_year  # Define current_vehicle_age here
        expected_lifespan = get_vehicle_lifespan(make, model)  # Also define expected_lifespan here
        
        for i, row in chart_df.iterrows():
            year_num = i + 1
            activities = row['Activities']
            cost = row['Maintenance Cost']
            vehicle_age_at_year = current_vehicle_age + year_num
            
            if activities and activities != 'None':
                # Split activities into a readable list
                activity_list = [activity.strip() for activity in activities.split(',')]
                
                # Create aging context
                aging_context = ""
                if vehicle_age_at_year > expected_lifespan:
                    aging_context = f" ⚠️ (Vehicle age: {vehicle_age_at_year} years - BEYOND {expected_lifespan}-year expected lifespan)"
                elif vehicle_age_at_year > expected_lifespan * 0.8:
                    aging_context = f" 🟡 (Vehicle age: {vehicle_age_at_year} years - {((vehicle_age_at_year/expected_lifespan)*100):.0f}% of expected lifespan)"
                
                st.markdown(f"**Year {year_num} - ${cost:,.0f}{aging_context}**")
                
                for activity in activity_list:
                    # Add emoji indicators for different types of maintenance
                    if any(keyword in activity for keyword in ['Major', 'Overhaul', 'Rebuild']):
                        st.markdown(f"  🚨 {activity}")
                    elif any(keyword in activity for keyword in ['Replacement', 'Service']):
                        st.markdown(f"  🔧 {activity}")
                    elif any(keyword in activity for keyword in ['Check', 'Inspection']):
                        st.markdown(f"  🔍 {activity}")
                    elif any(keyword in activity for keyword in ['Oil', 'Filter', 'Rotation']):
                        st.markdown(f"  🛠️ {activity}")
                    else:
                        st.markdown(f"  • {activity}")
                
                st.markdown("---")
            else:
                st.markdown(f"**Year {year_num} - ${cost:,.0f}**")
                st.markdown("  ✅ No scheduled major maintenance")
                st.markdown("---")

    # Vehicle aging analysis
    current_vehicle_age = datetime.now().year - model_year
    final_vehicle_age = current_vehicle_age + years
    expected_lifespan = get_vehicle_lifespan(make, model)
    
    if final_vehicle_age > expected_lifespan * 0.8:
        st.warning(f"""
        ⚠️ **Vehicle Aging Alert**: By year {years}, your {make} {model} will be {final_vehicle_age} years old, 
        which is {((final_vehicle_age/expected_lifespan)*100):.0f}% of its expected {expected_lifespan}-year lifespan. 
        Expect higher maintenance costs and potential major repairs.
        """)
    
    # Show cost escalation due to aging
    if years > 3:
        early_years_avg = chart_df['Maintenance Cost'].iloc[:3].mean()
        later_years_avg = chart_df['Maintenance Cost'].iloc[3:].mean() if len(chart_df) > 3 else early_years_avg
        
        if later_years_avg > early_years_avg * 1.2:  # 20% increase
            cost_increase = ((later_years_avg / early_years_avg - 1) * 100)
            st.info(f"""
            📈 **Aging Cost Impact**: Maintenance costs increase by {cost_increase:.0f}% in later years 
            (${early_years_avg:,.0f} early vs ${later_years_avg:,.0f} later) due to vehicle aging.
            """)

    # Calculate totals & percentages
    total = chart_df['Total Cost'].sum()
    avg_annual = total / years
    pct_inc = (avg_annual / gross) * 100
    avg_insurance = chart_df['Insurance Premium'].mean()

    # Financial Analysis
    st.header("💵 Financial Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cost Over Period", f"${total:,.0f}")
    with col2:
        st.metric("Average Annual Cost", f"${avg_annual:,.0f}")
    with col3:
        st.metric("% of Gross Income", f"{pct_inc:.1f}%")
    with col4:
        st.metric("Est. Annual Insurance", f"${avg_insurance:,.0f}")
    
    # Additional aging-related metrics
    if years >= 5:
        col1, col2, col3 = st.columns(3)
        with col1:
            total_maintenance = chart_df['Maintenance Cost'].sum()
            st.metric("Total Maintenance", f"${total_maintenance:,.0f}")
        with col2:
            total_depreciation = chart_df['Depreciation Cost'].sum()
            st.metric("Total Depreciation", f"${total_depreciation:,.0f}")
        with col3:
            final_value = chart_df['Car Value'].iloc[-1]
            st.metric("Final Vehicle Value", f"${final_value:,.0f}")
    
    # Lifetime value analysis
    if final_vehicle_age > expected_lifespan * 0.7:
        st.subheader("🕰️ Vehicle Lifespan Analysis")
        
        # Calculate remaining value proposition
        remaining_years = max(1, expected_lifespan - final_vehicle_age)
        cost_per_remaining_year = avg_annual
        
        if remaining_years <= 2:
            st.error(f"""
            🚨 **End-of-Life Alert**: Your vehicle will have only ~{remaining_years} year(s) of useful life remaining 
            after this forecast period. Consider replacement planning.
            """)
        elif remaining_years <= 5:
            st.warning(f"""
            ⚠️ **Replacement Planning**: Your vehicle will have ~{remaining_years} years of useful life remaining. 
            At ${cost_per_remaining_year:,.0f}/year, total remaining ownership cost could be 
            ${cost_per_remaining_year * remaining_years:,.0f}.
            """)
        
        # Replacement vs keep analysis
        if final_vehicle_age > expected_lifespan * 0.8:
            st.markdown("### 🤔 Keep vs Replace Analysis")
            
            # Estimate cost of similar replacement vehicle
            current_year = datetime.now().year
            replacement_cost = your_price * 1.2  # Assume 20% more for newer model
            final_value = chart_df['Car Value'].iloc[-1]
            
            net_replacement_cost = replacement_cost - final_value
            
            st.info(f"""
            **Replacement Consideration:**
            - **Current Vehicle Value (Year {years}):** ${final_value:,.0f}
            - **Estimated Replacement Cost:** ${replacement_cost:,.0f}
            - **Net Cost to Replace:** ${net_replacement_cost:,.0f}
            - **Benefit:** Reset to {expected_lifespan}-year lifespan with lower maintenance costs
            """)
            
            if net_replacement_cost < avg_annual * 2:
                st.success("💡 **Replacement may be cost-effective** - Net replacement cost is less than 2 years of current ownership costs")
            elif net_replacement_cost < avg_annual * 4:
                st.warning("⚠️ **Replacement worth considering** - Weigh reliability benefits vs costs")
            else:
                st.info("📊 **Keeping current vehicle may be economical** - High replacement cost relative to ownership costs")

    # Financial recommendations
    st.header("📈 Financial Recommendations")
    
    # Calculate loan payment for analysis
    r = irate / 100 if irate > 0 else 0
    if loan_amount > 0 and r > 0:
        monthly_loan_payment = (loan_amount * (r/12)) / (1 - (1 + r/12) ** (-lt_years * 12))
    elif loan_amount > 0:
        monthly_loan_payment = loan_amount / (lt_years * 12)
    else:
        monthly_loan_payment = 0
    
    annual_loan_payment = monthly_loan_payment * 12
    
    # Total transportation cost analysis (the real 10% rule)
    recommended_max_annual = gross * 0.10  # 10% of gross income for total transportation
    
    # Calculate costs without loan payments to see underlying vehicle costs
    annual_without_loan = avg_annual - annual_loan_payment if loan_amount > 0 else avg_annual
    
    # Format all numbers properly before display
    avg_annual_formatted = f"${avg_annual:,.0f}"
    recommended_max_formatted = f"${recommended_max_annual:,.0f}"
    
    if avg_annual > recommended_max_annual:
        overage_amount = avg_annual - recommended_max_annual
        st.warning(f"""
        ⚠️ **Transportation Budget Alert**
        
        Your annual ownership costs of **{avg_annual_formatted}** exceed the recommended maximum of **{recommended_max_formatted}** (10% of gross income).
        
        **You're ${overage_amount:,.0f} over the recommended budget.**
        """)
        
        # Provide specific advice based on what's driving the overage
        if loan_amount > 0:
            if annual_loan_payment > recommended_max_annual * 0.6:  # If loan is >60% of transport budget
                st.markdown("""
                **💡 Primary Issue:** Loan payments are consuming most of your transportation budget
                
                **Recommendations:**
                • Consider a less expensive vehicle to reduce monthly payments
                • Increase your down payment to lower the loan amount  
                • Extend the loan term to reduce monthly payments (but increase total interest)
                """)
            else:
                st.markdown("**💡 Primary Issue:** High ongoing ownership costs")
        
    else:
        savings_amount = recommended_max_annual - avg_annual
        savings_pct = (savings_amount / recommended_max_annual) * 100
        st.success(f"""
        ✅ **Excellent Budget Management**
        
        Your annual ownership costs of **{avg_annual_formatted}** are within the recommended range.
        
        **You have ${savings_amount:,.0f} ({savings_pct:.0f}%) remaining in your transportation budget.**
        """)
    
    # Purchase price analysis - focus on loan impact
    if loan_amount > 0:
        # If there's a loan, focus on payment affordability
        monthly_take_home = (gross * 0.75) / 12  # Estimate take-home as 75% of gross
        monthly_payment_pct = (monthly_loan_payment / monthly_take_home) * 100
        
        if monthly_payment_pct > 15:
            st.error(f"""
            🚨 **High Monthly Payment**
            
            Your car payment of **${monthly_loan_payment:,.0f}** represents **{monthly_payment_pct:.1f}%** of estimated take-home pay. 
            
            Financial experts recommend keeping car payments under 10-15% of take-home pay.
            """)
        elif monthly_payment_pct > 10:
            st.warning(f"""
            ⚠️ **Moderate Monthly Payment**
            
            Your car payment of **${monthly_loan_payment:,.0f}** is **{monthly_payment_pct:.1f}%** of estimated take-home pay. 
            
            This is manageable but limits flexibility for other goals.
            """)
        else:
            st.success(f"""
            ✅ **Affordable Monthly Payment**
            
            Your car payment of **${monthly_loan_payment:,.0f}** is **{monthly_payment_pct:.1f}%** of estimated take-home pay, leaving room for other financial priorities.
            """)
    else:
        # If paying cash, evaluate purchase price against income
        purchase_pct = (your_price / gross) * 100
        if purchase_pct > 50:
            st.error(f"""
            🚨 **Very High Purchase Price**
            
            Spending **${your_price:,.0f}** ({purchase_pct:.0f}% of annual income) on a vehicle may significantly impact your financial flexibility.
            """)
        elif purchase_pct > 25:
            st.warning(f"""
            ⚠️ **High Purchase Price**
            
            Spending **${your_price:,.0f}** ({purchase_pct:.0f}% of annual income) is substantial. Ensure this aligns with your financial goals.
            """)
        else:
            st.success(f"""
            ✅ **Reasonable Purchase Price**
            
            Spending **${your_price:,.0f}** ({purchase_pct:.0f}% of annual income) leaves room for other financial priorities.
            """)
    
    # Comprehensive breakdown
    st.info(f"""
    **📊 Transportation Budget Analysis**
    
    **Your Current Situation:**
    • **Total Annual Transportation Costs:** ${avg_annual:,.0f} ({pct_inc:.1f}% of income)
    • **Recommended Maximum:** ${recommended_max_annual:,.0f} (10% of income)
    • **Annual Loan Payments:** ${annual_loan_payment:,.0f} ({((annual_loan_payment)/gross)*100:.1f}% of income)
    • **Other Ownership Costs:** ${annual_without_loan:,.0f} (insurance, fuel, maintenance, etc.)
    
    **Financial Guidelines:**
    • **Total Transportation:** 10-15% of gross income maximum
    • **Monthly Car Payment:** 10-15% of monthly take-home pay maximum
    • **Cash Purchase:** Generally under 25% of annual income
    """)
    
    # Enhanced actionable suggestions if over budget
    if avg_annual > recommended_max_annual:
        st.markdown("**💡 Strategies to Bring Your Transportation Costs Within Budget:**")
        
        # Analyze what's driving the high costs
        avg_maintenance = chart_df['Maintenance Cost'].mean()
        avg_fuel = chart_df['Fuel/Electricity Cost'].mean()
        avg_insurance = chart_df['Insurance Premium'].mean()
        
        # Calculate what each component represents as % of the overage
        overage_amount = avg_annual - recommended_max_annual
        
        # Get current vehicle tier for recommendations
        current_tier = get_car_tier(make)
        
        # Priority recommendations based on biggest cost drivers
        st.markdown("### 🎯 Priority Actions (Biggest Impact):")
        
        # 1. Vehicle Choice Recommendations
        if avg_maintenance > recommended_max_annual * 0.15:  # If maintenance is >15% of recommended budget
            st.error(f"🔧 **Primary Issue: High Maintenance Costs** (${avg_maintenance:,.0f}/year)")
            
            if current_tier == 'Luxury':
                st.markdown("**💰 Vehicle Tier Recommendation: Switch to Midrange or Economy**")
                
                # Suggest specific alternatives
                economy_alternatives = []
                midrange_alternatives = []
                
                # Based on current vehicle type, suggest alternatives
                if model in ['MDX', 'X3', 'X5', 'GLC', 'RX', 'XC60']:  # Luxury SUVs
                    economy_alternatives = ["Honda CR-V", "Toyota RAV4", "Mazda CX-5"]
                    midrange_alternatives = ["Chevrolet Equinox", "Ford Escape", "Nissan Rogue"]
                elif model in ['C-Class', 'ES', 'TLX']:  # Luxury sedans
                    economy_alternatives = ["Honda Accord", "Toyota Camry", "Mazda 6"]
                    midrange_alternatives = ["Chevrolet Malibu", "Nissan Altima", "Hyundai Sonata"]
                elif model in ['911', 'Cayenne', 'NSX']:  # Luxury sports/premium
                    economy_alternatives = ["Honda Civic", "Toyota Corolla", "Mazda 3"]
                    midrange_alternatives = ["Ford Mustang", "Chevrolet Camaro", "Nissan Sentra"]
                
                if economy_alternatives:
                    st.success(f"✅ **Economy Alternatives**: {', '.join(economy_alternatives)}")
                    st.caption("💡 Economy brands typically have 40-60% lower maintenance costs")
                
                if midrange_alternatives:
                    st.info(f"📊 **Midrange Alternatives**: {', '.join(midrange_alternatives)}")
                    st.caption("💡 Midrange options balance features with 20-30% lower maintenance costs")
                
                # Calculate potential savings
                if current_tier == 'Luxury':
                    economy_savings = avg_maintenance * 0.45  # 45% savings switching to economy
                    midrange_savings = avg_maintenance * 0.25  # 25% savings switching to midrange
                    st.markdown(f"**Potential Annual Savings:**")
                    st.markdown(f"• Economy vehicle: **${economy_savings:,.0f}** less in maintenance")
                    st.markdown(f"• Midrange vehicle: **${midrange_savings:,.0f}** less in maintenance")
            
            elif current_tier == 'Midrange':
                st.markdown("**💰 Vehicle Tier Recommendation: Switch to Economy**")
                economy_alternatives = ["Honda Civic", "Toyota Corolla", "Honda Accord", "Toyota Camry", "Honda CR-V", "Toyota RAV4"]
                st.success(f"✅ **Economy Alternatives**: {', '.join(economy_alternatives[:3])}")
                economy_savings = avg_maintenance * 0.25
                st.markdown(f"**Potential Annual Savings: ${economy_savings:,.0f}** less in maintenance")
            
            else:  # Already economy
                st.markdown("**🔧 Consider Different Economy Models:**")
                st.markdown("• Honda and Toyota typically have the lowest maintenance costs")
                st.markdown("• Avoid luxury features that increase maintenance complexity")
        
        # 2. Loan/Purchase Price Recommendations
        if loan_amount > 0 and annual_loan_payment > recommended_max_annual * 0.5:
            st.error(f"💳 **Major Issue: Vehicle Too Expensive** (${annual_loan_payment:,.0f}/year in payments)")
            
            # Calculate target vehicle price
            target_annual_payment = recommended_max_annual * 0.4  # 40% of transport budget for payments
            if r > 0:
                target_loan_amount = target_annual_payment * (1 - (1 + r) ** -lt_years) / r
            else:
                target_loan_amount = target_annual_payment * lt_years
            
            target_vehicle_price = target_loan_amount + (your_price - loan_amount)  # Add down payment back
            
            st.markdown(f"**🎯 Target Vehicle Price: ${target_vehicle_price:,.0f}** (vs. current ${your_price:,.0f})")
            
            price_reduction_needed = your_price - target_vehicle_price
            st.markdown(f"**Need to reduce vehicle price by: ${price_reduction_needed:,.0f}**")
            
            # Specific actions
            st.markdown("**Immediate Actions:**")
            st.markdown(f"• Look for vehicles under **${target_vehicle_price:,.0f}**")
            st.markdown("• Increase down payment to reduce loan amount")
            st.markdown("• Consider certified pre-owned instead of new")
            st.markdown("• Extend loan term to reduce monthly payments (costs more long-term)")
        
        elif loan_amount == 0 and your_price > gross * 0.25:
            st.warning(f"💰 **Cash Purchase Too High** (${your_price:,.0f} = {(your_price/gross)*100:.0f}% of income)")
            target_cash_price = gross * 0.20  # 20% of income
            st.markdown(f"**🎯 Target Cash Price: ${target_cash_price:,.0f}**")
        
        # 3. Fuel/Electricity Cost Recommendations
        if avg_fuel > recommended_max_annual * 0.12:  # If fuel is >12% of recommended budget
            st.warning(f"⛽ **High Fuel/Energy Costs** (${avg_fuel:,.0f}/year)")
            
            if is_ev:
                st.markdown("**EV Optimization Strategies:**")
                st.markdown("• Switch to 'Home Only' charging if possible")
                st.markdown("• Contact utility about EV time-of-use rates")
                st.markdown("• Reduce daily driving or consider a more efficient EV")
                
                # Show potential savings from better charging
                if custom_rates_dict.get('use_custom', False):
                    current_rates = custom_rates_dict
                else:
                    current_rates = None
                    
                home_only_cost = calculate_ev_electricity_cost(make, model, avg_mpy, state, 'home', current_rates)
                if avg_fuel > home_only_cost:
                    charging_savings = avg_fuel - home_only_cost
                    st.success(f"💡 **Home-only charging could save ${charging_savings:,.0f}/year**")
            else:
                # Suggest more fuel-efficient alternatives
                current_mpg = average_mpg.get(make, {}).get(model, mpg)
                st.markdown("**Fuel Efficiency Recommendations:**")
                
                if current_mpg < 25:
                    efficient_alternatives = ["Honda Civic (32 MPG)", "Toyota Corolla (33 MPG)", "Nissan Sentra (29 MPG)", "Hyundai Elantra (33 MPG)"]
                    st.success(f"✅ **High-Efficiency Alternatives**: {', '.join(efficient_alternatives[:2])}")
                    
                    # Calculate fuel savings with better MPG
                    better_mpg = 32  # Average of efficient cars
                    fuel_price = get_fuel_price(state, make, model, custom_fuel_price if 'custom_fuel_price' in locals() else None)
                    better_fuel_cost = (avg_mpy / better_mpg) * fuel_price
                    fuel_savings = avg_fuel - better_fuel_cost
                    st.markdown(f"**Potential Annual Savings: ${fuel_savings:,.0f}** with 32 MPG vehicle")
                
                elif current_mpg < 30:
                    st.markdown("• Consider hybrid options (Toyota Prius: 50+ MPG)")
                    st.markdown("• Reduce annual mileage if possible")
                
                st.markdown("**Immediate Actions:**")
                st.markdown("• Improve driving habits (avoid aggressive acceleration)")
                st.markdown("• Combine trips to reduce total miles")
                st.markdown("• Consider work-from-home to reduce commuting")
        
        # 4. Insurance Cost Recommendations
        if avg_insurance > recommended_max_annual * 0.08:  # If insurance is >8% of recommended budget
            st.warning(f"🛡️ **High Insurance Costs** (${avg_insurance:,.0f}/year)")
            st.markdown("**Insurance Reduction Strategies:**")
            st.markdown("• Shop with 3+ insurers annually (can save 20-40%)")
            st.markdown("• Increase deductibles to lower premiums")
            st.markdown("• Consider usage-based insurance if you drive less")
            if msrp and msrp > 50000:
                st.markdown("• **Major Impact**: Choose vehicle under $50,000 to avoid luxury premium")
            if avg_mpy > 15000:
                st.markdown("• Reduce annual mileage below 15,000 if possible")
        
        st.markdown("---")
        
        # Overall Strategy Summary
        st.markdown("### 📋 Complete Budget Strategy:")
        
        # Create a priority-ordered action plan
        action_plan = []
        
        # Calculate impact of each change
        if current_tier == 'Luxury' and avg_maintenance > recommended_max_annual * 0.15:
            maintenance_savings = avg_maintenance * 0.45  # Economy car savings
            action_plan.append(f"**1. Switch to Economy Vehicle** → Save ${maintenance_savings:,.0f}/year")
        
        if loan_amount > 0 and annual_loan_payment > recommended_max_annual * 0.5:
            target_payment = recommended_max_annual * 0.4
            payment_savings = annual_loan_payment - target_payment
            action_plan.append(f"**2. Reduce Vehicle Price** → Save ${payment_savings:,.0f}/year in payments")
        
        if avg_fuel > recommended_max_annual * 0.12 and not is_ev:
            fuel_savings = avg_fuel * 0.30  # Estimate 30% savings with efficient car
            action_plan.append(f"**3. Choose Fuel-Efficient Vehicle** → Save ${fuel_savings:,.0f}/year")
        
        if avg_insurance > recommended_max_annual * 0.08:
            insurance_savings = avg_insurance * 0.25  # 25% savings from shopping/optimization
            action_plan.append(f"**4. Optimize Insurance** → Save ${insurance_savings:,.0f}/year")
        
        # Display the action plan
        for action in action_plan:
            st.markdown(action)
        
        # Calculate total potential savings
        if action_plan:
            total_potential_savings = 0
            if current_tier == 'Luxury':
                total_potential_savings += avg_maintenance * 0.45
            if loan_amount > 0 and annual_loan_payment > recommended_max_annual * 0.5:
                total_potential_savings += annual_loan_payment - (recommended_max_annual * 0.4)
            if avg_fuel > recommended_max_annual * 0.12 and not is_ev:
                total_potential_savings += avg_fuel * 0.30
            if avg_insurance > recommended_max_annual * 0.08:
                total_potential_savings += avg_insurance * 0.25
            
            if total_potential_savings > overage_amount:
                st.success(f"🎉 **Following these recommendations could save ${total_potential_savings:,.0f}/year** - enough to get within your ${recommended_max_annual:,.0f} budget!")
            else:
                st.info(f"💡 **These strategies could save ${total_potential_savings:,.0f}/year** - getting you closer to your budget goal")
        
        # Quick wins section
        st.markdown("### ⚡ Quick Wins (Start Today):")
        quick_wins = [
            "📞 Get 3 insurance quotes (potential 20-40% savings)",
            "📱 Download a fuel tracking app to monitor consumption",
            "🚗 Research reliable economy vehicles in your area",
            "💰 Calculate how much down payment you could add",
            "📊 Review your actual driving needs vs. wants"
        ]
        
        for win in quick_wins:
            st.markdown(f"• {win}")
    
    else:
        # If within budget, still provide optimization suggestions
        current_tier = get_car_tier(make)  # Define current_tier here too for the else block
        
        st.markdown("### 💡 Optimization Opportunities:")
        
        savings_amount = recommended_max_annual - avg_annual
        st.markdown(f"You're ${savings_amount:,.0f} under budget. Consider these optimizations:")
        
        optimization_tips = []
        
        if current_tier == 'Luxury':
            optimization_tips.append("• You chose a luxury vehicle - ensure it aligns with your long-term financial goals")
        
        if is_ev:
            optimization_tips.append("• Maximize home charging to reduce electricity costs")
            optimization_tips.append("• Look into solar panels to further reduce charging costs")
        else:
            optimization_tips.append("• Consider a hybrid or electric vehicle for even lower operating costs")
        
        optimization_tips.append("• Use the extra budget room to build an emergency fund")
        optimization_tips.append("• Consider increasing retirement contributions with the saved money")
        
        for tip in optimization_tips:
            st.markdown(tip)
    
    # Electric vehicle specific analysis
    if is_ev:
        st.header("🔋 Electric Vehicle Cost Analysis")
        
        # Get custom rates if available from the UI
        if custom_rates_dict.get('use_custom', False):
            custom_rates = {
                'residential': custom_rates_dict['residential'],
                'ev_rate': custom_rates_dict['ev_rate'],
                'public': custom_rates_dict['public'],
                'has_ev_rate': custom_rates_dict['has_ev_rate']
            }
        else:
            custom_rates = None
        
        # Calculate detailed electricity costs
        annual_electricity = calculate_ev_electricity_cost(make, model, avg_mpy, state, charging_pref, custom_rates)
        
        # EV vs Gas comparison
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Annual Electricity Cost", f"${annual_electricity:,.0f}")
            
        with col2:
            # Calculate equivalent gas cost for comparison
            equivalent_gas_mpg = 25  # Average ICE vehicle
            gas_price = state_fuel_prices.get(state, {'regular': 3.50})['regular']
            equivalent_gas_cost = (avg_mpy / equivalent_gas_mpg) * gas_price
            annual_savings = equivalent_gas_cost - annual_electricity
            st.metric("Annual Fuel Savings vs Gas", f"${annual_savings:,.0f}")
        
        # Multi-year savings
        total_electricity = chart_df['Fuel/Electricity Cost'].sum()
        total_gas_equivalent = equivalent_gas_cost * years
        total_savings = total_gas_equivalent - total_electricity
        
        st.success(f"""
        **⚡ Electric Vehicle Benefits Over {years} Years:**
        - **Total Electricity Cost:** ${total_electricity:,.0f}
        - **Equivalent Gas Cost:** ${total_gas_equivalent:,.0f}
        - **Total Fuel Savings:** ${total_savings:,.0f}
        - **Lower maintenance costs** (no oil changes, fewer moving parts)
        - **Environmental benefits** and potential tax incentives
        """)
        
        # Charging cost breakdown
        st.subheader("📊 Charging Cost Breakdown")
        
        # Calculate costs for different charging methods using custom rates if available
        home_cost = calculate_ev_electricity_cost(make, model, avg_mpy, state, 'home', custom_rates)
        public_cost = calculate_ev_electricity_cost(make, model, avg_mpy, state, 'public', custom_rates)
        mixed_cost = calculate_ev_electricity_cost(make, model, avg_mpy, state, 'mixed', custom_rates)
        
        charging_df = pd.DataFrame({
            'Charging Method': ['Home Only', 'Public Only', 'Mixed (70% Home)'],
            'Annual Cost': [home_cost, public_cost, mixed_cost],
            'Cost per Mile': [home_cost/avg_mpy, public_cost/avg_mpy, mixed_cost/avg_mpy]
        })
        
        st.dataframe(charging_df, use_container_width=True)
        
        # Show rate information being used
        if custom_rates:
            st.info(f"""
            **📊 Using Your Custom Rates:**
            - Home Rate: ${custom_rates['residential']:.3f}/kWh
            - EV Rate: ${custom_rates['ev_rate']:.3f}/kWh {'(Time-of-Use)' if custom_rates['has_ev_rate'] else '(Same as residential)'}
            - Public Rate: ${custom_rates['public']:.2f}/kWh
            """)
        else:
            default_rates = get_ev_charging_info(make, model, state)
            if default_rates:
                st.info(f"""
                **📊 Using Default {state} Rates:**
                - Residential Rate: ${default_rates['base_rate']:.3f}/kWh
                - EV Rate: ${default_rates['ev_rate']:.3f}/kWh
                - Public Rate: $0.35/kWh (National Average)
                """)
        
        # EV efficiency details
        ev_info = get_ev_charging_info(make, model, state)
        if ev_info:
            st.subheader("🔌 Vehicle Efficiency Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                kwh_per_year = avg_mpy / ev_info['efficiency']
                st.metric("Annual kWh Usage", f"{kwh_per_year:,.0f} kWh")
                
            with col2:
                cost_per_mile = annual_electricity / avg_mpy
                st.metric("Cost per Mile", f"${cost_per_mile:.3f}")
                
            with col3:
                full_charges_per_year = avg_mpy / ev_info['range']
                st.metric("Full Charges/Year", f"{full_charges_per_year:.0f}")
        
        st.caption("""
        **EV Cost Calculation Notes:**
        - Home charging includes ~10-12% charging losses
        - Public charging includes ~15-18% charging losses  
        - Time-of-use rates applied where available
        - Costs vary significantly by charging habits and local electricity rates
        - Custom rates override default state averages for more accurate calculations
        """)
    
    else:
        # Gas vehicle fuel analysis
        total_fuel_cost = chart_df['Fuel/Electricity Cost'].sum()
        gallons_per_year = avg_mpy / mpg
        total_gallons = gallons_per_year * years
        
        st.header("⛽ Fuel Cost Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Annual Fuel Cost", f"${total_fuel_cost/years:,.0f}")
            st.metric("Gallons per Year", f"{gallons_per_year:,.0f}")
        with col2:
            st.metric(f"Total Fuel Cost ({years} years)", f"${total_fuel_cost:,.0f}")
            st.metric("Total Gallons Used", f"{total_gallons:,.0f}")
        
        fuel_price = get_fuel_price(state, make, model, custom_fuel_price)
        st.caption(f"Based on {fuel_type} gasoline at ${fuel_price:.2f}/gallon")
    
    # Insurance information
    st.markdown("""
    **Insurance Estimate Details:**  
    Our estimate starts with a $1,200 base premium for a safe driver and adjusts for:
    - Age (higher premiums for drivers under 25)
    - Driving experience (discounts after 10+ years)  
    - Vehicle value (higher premiums for cars over $50,000)
    - Annual mileage and state cost factors
    
    Actual premiums vary significantly based on your driving record, coverage levels, and insurance provider.
    """)