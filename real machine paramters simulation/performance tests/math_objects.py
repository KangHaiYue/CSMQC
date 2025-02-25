from numpy import array, kron, sqrt, exp, pi, sin, cos, eye

pauli = {'I': array([[1, 0], [0, 1]], dtype=complex),
         'X': array([[0, 1], [1, 0]], dtype=complex),
         'Y': array([[0, -1j], [1j, 0]], dtype=complex),
         'Z': array([[1, 0], [0, -1]], dtype=complex)}

pauli_product = {'II':(1,'I'), 'IX':(1,'X'), 'IY':(1,'Y'), 'IZ':(1,'Z'),
                 'XI':(1,'X'), 'XX':(1,'I'), 'XY':(1j,'Z'), 'XZ':(-1j,'Y'),
                'YI':(1,'Y'), 'YX':(-1j,'Z'), 'YY':(1,'I'), 'YZ':(1j,'X'),
                'ZI':(1,'Z'), 'ZX':(1j,'Y'), 'ZY':(-1j,'X'), 'ZZ':(1,'I')}

H = array([[1, 1], [1, -1]], dtype=complex)*(1/sqrt(2))
CNOT = array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
CNOT_reserved = array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)
S = array([[1, 0], [0, 1j]], dtype=complex)
T = array([[1, 0], [0, exp(1j*pi/4)]], dtype=complex)
Tdg = array([[1, 0], [0, exp(-1j*pi/4)]], dtype=complex)
X = array([[0, 1], [1, 0]], dtype=complex)
Xpi2 = array([[1, 1j], [1j, 1]], dtype=complex)*(1/sqrt(2))
CCZ = eye(2**3, dtype=complex)
CCZ[-1][-1] = -1

POPCOUNT_TABLE16 = [0, 1] * 2**15
for index in range(2, len(POPCOUNT_TABLE16)):  # 0 and 1 are trivial
    POPCOUNT_TABLE16[index] += POPCOUNT_TABLE16[index >> 1]

def hamming_weight(n):
    """return the Hamming weight of an integer (check how many '1's for an integer after converted to binary)

    Args:
        n (int): any integer

    Returns:
        int: number of ones of the integer n after converted to binary
    """
    c = 0
    while n:
        c += POPCOUNT_TABLE16[n & 0xffff]
        n >>= 16
    return c

def pauli_n(basis_str):
    """Calculate kronecker tensor product sum of basis from basis string"""
    
    M = pauli[basis_str[0]]
    try:
        for basis in basis_str[1:]:
            M_new = kron(M, pauli[basis])
            M = M_new
    except: pass # Single basis case
    
    return M 

def bit_str_list(n):
    """Create list of all n-bit binary strings"""
    return [format(i, 'b').zfill(n) for i in range(2**n)]

def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  sqrt(x*x + y*y + z*z)
    theta   =  arccos(z/r)
    phi     =  arctan2(y,x)
    return [r,theta,phi]

def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1]
    phi     = rthetaphi[2]
    x = r * sin( theta ) * cos( phi )
    y = r * sin( theta ) * sin( phi )
    z = r * cos( theta )
    return [x,y,z]

def distance_on_sphere(coord_a, coord_b):
    #coordinates must have norm = 1
    #coord_a /= norm(coord_a)
    #coord_b /= norm(coord_b)
    return arccos(np.dot(coord_a, coord_b))
    
def total_distance(xyz, coordiantes):
    #xyz /= norm(xyz)
    distance = 0
    for coordinate in coordiantes:
        distance += distance_on_sphere(xyz, coordinate)
    
    return distance

def con(xyz):
    return xyz[0]**2 + xyz[1]**2 + xyz[2]**2 - 1

edges27 = [(0, 1),
           (1, 2),
           (2, 3),
           (1, 4),
           (3, 5),
           (4, 7),
           (5, 8),
           (6, 7),
           (8, 9),
           (7, 10),
           (8, 11),
           (10, 12),
           (11, 14),
           (12, 13),
           (13, 14),
           (12, 15),
           (14, 16),
           (15, 18),
           (16, 19),
           (17, 18),
           (19, 20),
           (18, 21),
           (19, 22),
           (21, 23),
           (22, 25),
           (23, 24),
           (24, 25),
           (25, 26)]