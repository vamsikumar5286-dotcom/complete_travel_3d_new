from main import load_models, predict_transport, TransportInput

# Ensure models are loaded into main's globals
load_models()

tests = [
    TransportInput(transport_type='train', distance_km=500, city='Bengaluru', passengers=1, train_class='ordinary passenger'),
    TransportInput(transport_type='bus', distance_km=450, city='Mumbai', passengers=1, train_class=None),
    TransportInput(transport_type='flight', distance_km=1200, city='Delhi', passengers=1, train_class=None),
]

for t in tests:
    print('\n--- Testing:', t.transport_type)
    res = predict_transport(t)
    print(res)
