def somme(n):
    s = 0
    while n != 0:
        n = n % 10
        s += n
    return s

print(somme(123))