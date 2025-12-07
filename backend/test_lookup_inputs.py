import main
main.load_models()
lk=main.transport_model['lookup']
inputs = ['ordinary passenger','Ord_Passenger','ordinary',' ordinary passenger ','ORDINARY PASSENGER']
for s in inputs:
    print(repr(s), '->', lk.get_base_price(s))

# distance lookup
print('\nDistance lookups:')
for d in [10,50,100,200,500]:
    print('dist',d,'->', lk.get_distance_price('Ord_Passenger', d))
