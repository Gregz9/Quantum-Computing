import itertools

arr = [0, 1]

for bits in itertools.product(arr, repeat=2):
    print(bits)
