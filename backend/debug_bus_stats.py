import pandas as pd
import numpy as np

# Load bus dataset
df = pd.read_csv('indian_bus_fare_dataset.csv')

print('='*60)
print('BUS DATASET ANALYSIS')
print('='*60)
print(f'\nDataset Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')

print('\n' + '='*60)
print('OVERALL FARE STATISTICS')
print('='*60)
print(df['Fare Price (INR)'].describe())

print('\n' + '='*60)
print('STATISTICS BY BUS TYPE')
print('='*60)
for bus_type in df['Bus Type'].unique():
    subset = df[df['Bus Type'] == bus_type]['Fare Price (INR)']
    print(f'\n{bus_type}:')
    print(f'  Count: {len(subset)}')
    print(f'  Mean: ₹{subset.mean():.2f}')
    print(f'  Median: ₹{subset.median():.2f}')
    print(f'  Min: ₹{subset.min():.2f}')
    print(f'  Max: ₹{subset.max():.2f}')

print('\n' + '='*60)
print('STATISTICS BY DISTANCE (APPROXIMATED FROM DURATION)')
print('='*60)
# Estimate distance from duration (assuming ~60km/hour average)
df['Estimated_Distance'] = df['Duration (hours)'] * 60

distance_bins = [0, 100, 300, 500, 1000, np.inf]
distance_labels = ['0-100km', '100-300km', '300-500km', '500-1000km', '1000+km']
df['Distance_Bin'] = pd.cut(df['Estimated_Distance'], bins=distance_bins, labels=distance_labels)

for distance_range in distance_labels:
    subset = df[df['Distance_Bin'] == distance_range]['Fare Price (INR)']
    if len(subset) > 0:
        print(f'\n{distance_range}:')
        print(f'  Count: {len(subset)}')
        print(f'  Mean: ₹{subset.mean():.2f}')
        print(f'  Median: ₹{subset.median():.2f}')
        print(f'  Min: ₹{subset.min():.2f}')
        print(f'  Max: ₹{subset.max():.2f}')

print('\n' + '='*60)
print('SAMPLES OF 500KM JOURNEYS (estimated)')
print('='*60)
samples_500km = df[(df['Estimated_Distance'] >= 450) & (df['Estimated_Distance'] <= 550)]
print(f'Found {len(samples_500km)} records in 450-550km range')
if len(samples_500km) > 0:
    print(samples_500km[['Agency', 'Source', 'Destination', 'Bus Type', 'Duration (hours)', 'Estimated_Distance', 'Fare Price (INR)']].head(10))

print('\n' + '='*60)
print('VOLVO BUSES - FARE STATISTICS')
print('='*60)
volvo_df = df[df['Bus Type'] == 'Volvo']['Fare Price (INR)']
print(f'Volvo Count: {len(volvo_df)}')
print(f'Volvo Mean: ₹{volvo_df.mean():.2f}')
print(f'Volvo Median: ₹{volvo_df.median():.2f}')
print(f'Volvo Min: ₹{volvo_df.min():.2f}')
print(f'Volvo Max: ₹{volvo_df.max():.2f}')
