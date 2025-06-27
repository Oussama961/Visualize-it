def saisir(L, C):
    A = []
    for i in range(L):
        T = []
        for j in range(C):
            T.append(float(input(f"Entrer l'element ({i},{j}): ")))
        A.append(T)
    return A

def affichage(A, L, C):
    for i in range(L):
        for j in range(C):
            print(A[i][j])

def somme_deux_matrice(A, B, L,C):
    for i in range(L):
        for j in range(C):
            A[i][j] = A[i][j]+B[i][j]
    return A
 
def produit_deux_matrice(A, B, L,C):
    m = len(B[0])
    p=len(B)
    P = [[0]*L for i in range(C)] 
    for i in range(L):
        for j in range(C):
            for k in range(p):
                P[i][j] += A[i][k] * B[k][j]
    return P


def main():
    l,c, i = int(input("Dimensions du matrice: (Ligne) : ")), int(input("Dimensions du matrice: (Column) : ")), 1
    print("Donner les valeurs des deux matrices :\n Matrice 1 (Notee: A) :")
    A = saisir(l,c)
    print("Donner les valeurs des deux matrices :\n Matrice 2 (Notee: B) :")
    B = saisir(l,c)

    P= produit_deux_matrice(A, B, l,c)
    
    S = somme_deux_matrice(A, B, l,c)
    
    
    des = str(input("Resultat ce forme d'un somme ou produit de les deux matrices.\n(S): pour somme.\n(P) pour produit.\n(Clique entrer pour les deux).\Processing...: "))

    if des == "S":
        print('La somme des deux matrices est:')
        print(S)
        re = input("Exiter le programme?.. (Y/N) : ")
        if re == "Y" or 'y':
            return
        else:
            print("_____ (...) _____")

        main()
    elif des == "P":
        print('Le produit des deux matrices est:')
        print(P)
        main()
    else:
        print('La somme des deux matrices est:')
        print(S)
        print('Le produit des deux matrices est:')
        print(P)
        main()


if __name__ == "__main__":
    main()

