import numpy as np
t = int(input())
start_state = int(input())
end_state = int(input())
n = int(input("no. of rows?"))
matrix = []
for i in range(n):
    a = []
    for j in range (n):
        entry = float(input())
        a.append(entry)
    matrix.append(a)
P = np.zeros(n)
P[start_state-1] = 1
for j in range(t):
    P = np.dot(matrix, P)
print (P[end_state-1])


                                                                                                                                                                                                                    


        