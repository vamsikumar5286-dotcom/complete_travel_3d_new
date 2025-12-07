import main
main.load_models()
lookup = main.transport_model.get('lookup')
print('BASE_MAP_KEYS sample:', list(lookup.base_map.keys())[:30])
print('BASE_MAP contains Ord_Passenger?', 'Ord_Passenger' in lookup.base_map)
print('Lowercase keys sample:', [k.lower() for k in list(lookup.base_map.keys())[:30]])
print('\nDistance columns sample:', list(lookup.distance_df.columns)[:30] if lookup.distance_df is not None else None)
