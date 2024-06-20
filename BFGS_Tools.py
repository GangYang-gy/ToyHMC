import numpy as np


def update_PQUT(x_mem, grad_mem, U_mem, B1):

    Nq = np.shape(x_mem)[1]

    # firstly sort previous {xi} in ascending order with respect to its log function    
    o = np.argsort(U_mem)

    S = np.array([])
    Y = np.array([])
    v_last = 0
    v_next = 1
    init = True
    while v_next < len(o):
        sk = x_mem[v_next, :] - x_mem[v_last, :]  #s_k = x_{k+1} - x_k, vector
        yk = grad_mem[v_next, :] - grad_mem[v_last,:]  #y_k = f'_{k+1} - f'_k
        #sk = x_mem[o[v_next], :] - x_mem[o[v_last], :]  #s_k = x_{k+1} - x_k, vector
        #yk = grad_mem[o[v_next], :] - grad_mem[o[v_last],:]  #y_k = f'_{k+1} - f'_k
        if np.dot(sk, yk.T) > 0:
            v_last = v_next
            v_next = v_next + 1
            if init:
                S = np.hstack((S.copy(),sk.T))
                Y = np.hstack((Y.copy(),yk.T))
                init = False
            else:
                S = np.vstack((S.copy(),sk.T))
                Y = np.vstack((Y.copy(),yk.T))
        else:
            v_next += 1

    vn = np.shape(S)[0]
    P = np.zeros((Nq, vn))
    Q = np.zeros((Nq, vn))
    U = np.array([])
    T = np.array([])

    init =True
    for k in range(vn):
        syk = np.dot(S[k,:].T, Y[k,:])
        Bsk = Bz_product(U, T, B1, S[k,:])
        sBsk = np.dot(S[k,:].T, Bsk)

        P[:, k] = S[k,:] / syk
        Q[:, k] = np.dot(np.sqrt(syk/sBsk), Bsk) + Y[k,:]    
        t_k = S[k,:]/sBsk
        u_k = np.dot(np.sqrt(sBsk/syk), Y[k,:])+Bsk
        if init:
            T= np.hstack((T.copy(),t_k))
            U= np.hstack((U.copy(),u_k))
            init = False
        else:
            T= np.vstack((T.copy(),t_k)) #np.vstack((T.copy(),t_k))
            U= np.vstack((U.copy(),u_k))   #dimension: Vn x Nq
    return P, Q, U, T


def Bz_product(U, T, B1, z):
    #B = CC^T
    if U.size == 0:
        Bz = np.dot(B1, z)
        return Bz
    # compute C^T s
    CTz = CTz_product(U, T, B1, z)
    # compute Ns = CC^T s
    Bz = Cz_product(U, T, B1, CTz)
    return Bz
        
def CTz_product(U, T, B1, z):
    # compute C^T * sk
    C1 = np.sqrt(B1)
    Csk = z
    try:
        m = np.shape(U)[1]
        m = np.shape(U)[0]
    except:
        m = 1

    if m==1:
        Csk = Csk - np.dot(np.dot(U.T, Csk), T)
    else:
        for k in range(m-1, -1, -1):
        #for k in reversed(range(m))
            Csk = Csk - np.dot(np.dot(U[k,:].T, Csk), T[k,:])
            #print(m,np.shape(U), np.shape(Csk), np.shape(T))
           # Csk = Csk - np.dot(np.dot(U[:, k].T, Csk), T[:, k])
    
    CTz = np.dot(C1, Csk)
    return CTz

def Cz_product(U, T, B1, z):
    # compute C * CTs = C * C^T * sk
    # or compute C * z, where z ~ N(0,I)
    C1 = np.sqrt(B1)
    CCTsk = np.dot(C1, z)
    try:
        m = np.shape(U)[1]
        m = np.shape(U)[0]
    except:
        m = 1

    if m == 1:
        CCTsk = CCTsk - np.dot(np.dot(T.T, CCTsk), U)
    else:
        for k in range(m):
            CCTsk = CCTsk - np.dot(np.dot(T[k,:].T, CCTsk), U[k,:])   # Ck - uk * tk^T *Ck
   
    CCTz = CCTsk
    return CCTz


def Hz_product(P, Q, B1, z):
    # H = SS^T
    #if P.size == 0:
     #   Hz = np.linalg.inv(B1) @ z
    # compute S^Tz
    STz = STz_product(P, Q, B1, z)
    # compute SS^Tz
    Hz = Sz_product(P, Q, B1, STz)
    return Hz

def STz_product(P, Q, B1, z):
    # compute S^Tz
    S1 = np.diag(np.sqrt(np.diag(B1)))
    m = P.shape[1]
    Siz = z.copy()
    for i in reversed(range(m)):
        Siz -= np.dot(np.dot(P[:, i].T, Siz), Q[:, i])

    STz = np.dot(S1, Siz)
    return STz

def Sz_product(P, Q, B1, z):
    S1 = np.diag(np.sqrt(np.diag(B1)))
    m = P.shape[1]
    Siz = np.dot(S1, z)
    for i in range(m):
        Siz -= np.dot(np.dot(Q[:, i].T, Siz), P[:, i])
    Sz = Siz
    return Sz