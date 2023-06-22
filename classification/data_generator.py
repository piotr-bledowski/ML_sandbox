import random

def generateLinearData(filename: str, separable: bool = False, n_samples: int = 100, x_dim: int = 2, scale: int = 10):
    f = open(filename + '.csv', 'w')
    n_neg = n_samples // 2

    # headers
    f.write(','.join(f'x{i}' for i in range(1, x_dim+1)) + ',y\n')

    if separable:
        for _ in range(n_neg):
            x = [random.random()*scale for __ in range(x_dim)]
            while sum(x) > (x_dim-1)*scale:
                x = [random.random() * scale for __ in range(x_dim)]
            f.write(','.join(str(n) for n in x) + ',-1\n')

        for _ in range(n_samples - n_neg):
            x = [random.random() * scale for __ in range(x_dim)]
            while sum(x) < (x_dim-1)*scale + scale / 4:
                x = [random.random() * scale for __ in range(x_dim)]
            f.write(','.join(str(n) for n in x) + ',1\n')
    else:
        for _ in range(n_neg):
            x = [random.random() * scale for __ in range(x_dim)]
            while sum(x) > (x_dim - 1) * scale:
                x = [random.random() * scale for __ in range(x_dim)]
            f.write(','.join(str(n) for n in x) + ',-1\n')

        for _ in range(n_samples - n_neg):
            x = [random.random() * scale for __ in range(x_dim)]
            while sum(x) < (x_dim - 1) * scale - scale / 4:
                x = [random.random() * scale for __ in range(x_dim)]
            f.write(','.join(str(n) for n in x) + ',1\n')

    f.close()