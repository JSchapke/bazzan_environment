
#F1 = [0.1 * i for i in range(1, 5)]
F1 = [0.4]
#F2 = [0.1 * i for i in range(1, 5)]
F2 = [0.2]
#F3 = [0.1 * i for i in range(5, 7)]
F3 = [0.5]

#V1 = range(100, 100, 10)
#V2 = range(100, 160, 10)
V1 = [100]
V2 = [130, 120]
C = 100

def so(c1, v1, v2, f1, f2, f3):
    so = 200 * (100*f1) + 100 * -(100*f2) + 200 * -c1 + 100 * v2 + 200 * v1
    print('SO:', so)
    print('SO / 600:', so / 300)

def check1(c1, v1, v2, f1, f2, f3):
    costA = 100 * (100*f1) + 100 * -(100*f2) \
            + 100 * -c1 + 100 * v2 + 100 * v1

    costB = 1 * -(1 * f3) + 101 * -(101 * f2) + 99 * (99 * f1) \
            + 99 * -c1 + 101 * v2 + 99 * v1

    return costB < costA


def check2(c1, v1, v2, f1, f2, f3):
    costA = -c1 + (100 * f1) + v1

    costB = -(1 * f3) -(101 * f2) + v2

    return costB > costA

def check(f1, f2, f3, v1, v2, c1):
    ch2 = check2(c1, v1, v2, f1, f2, f3)
    ch1 = check1(c1, v1, v2, f1, f2, f3)

    if ch1 and ch2:
        print('Checks:', f1, f2, f3, v1, v2, c1)
        so(c1, v1, v2, f1, f2, f3)


for f1 in F1:
    for f2 in F2:
        for f3 in F3:
            for v2 in V2:
                for v1 in V1:
                    check(f1, f2, f3, v1, v2, C)

