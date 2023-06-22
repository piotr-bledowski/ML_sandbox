import random

def generateLinearData(filename: str, separable: bool = False, n_samples: int = 100, x_dim: int = 2, scale: int = 10):
    f = open(filename, 'w')
    n_neg = n_samples // 2
    if separable:
        for _ in range(n_neg):
            x = [random.random()*scale for __ in range(x_dim)]
            while sum(x) > (x_dim-1)*scale:
                x = [random.random() * scale for __ in range(x_dim)]
            f.write(','.join(str(n) for n in x) + ',-1')

        for _ in range(n_samples - n_neg):
            x = [random.random()*scale + scale + scale / 4 for __ in range(x_dim)]
            while sum(x) < (x_dim-1)*scale + scale / 4:
                x = [random.random() * scale + scale + scale / 4 for __ in range(x_dim)]
            f.write(','.join(str(n) for n in x) + ',1')
    else:
        for _ in range(n_neg):
            x = [random.random() * scale for __ in range(x_dim)]
            while sum(x) > (x_dim - 1) * scale:
                x = [random.random() * scale for __ in range(x_dim)]
            f.write(','.join(str(n) for n in x) + ',-1')

        for _ in range(n_samples - n_neg):
            x = [random.random() * scale + scale - scale / 4 for __ in range(x_dim)]
            while sum(x) < (x_dim - 1) * scale - scale / 4:
                x = [random.random() * scale + scale - scale / 4 for __ in range(x_dim)]
            f.write(','.join(str(n) for n in x) + ',1')

    f.close()