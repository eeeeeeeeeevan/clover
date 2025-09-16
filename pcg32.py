import numpy as np

PCG32_INC = 105
PCG32_CONST = 0x5851F42D4C957F2D

def pcg32rand(st):
    oldst = st[0]
    st[0] = (oldst * PCG32_CONST + (PCG32_INC | 1)) & 0xFFFFFFFFFFFFFFFF
    xsh = ((oldst >> 18) ^ oldst) >> 27
    rot = oldst >> 59
    return ((xsh >> rot) | (xsh << ((-rot) & 31))) & 0xFFFFFFFF

def mathrand(st, lo, up):
    ul = up - lo
    x = (ul + 1) * pcg32rand(st)
    return lo + ((x >> 32) & 0xFFFFFFFF)

def pcg32seed(st, sd):
    st[0] = (PCG32_CONST * sd + 0x399D2694695129DE) & 0xFFFFFFFFFFFFFFFF

def xorsh(n):
    return ((n >> 18) ^ n) & 0xFFFFFFFFFFFFFFFF

def unxorsh(xsh):
    st = xsh & (~((1 << 46) - 1))
    for i in range(63 - 18, -1, -1):
        upb = (st >> (i + 18)) & 1
        cur = (xsh >> i) & 1
        nb = upb ^ cur
        st |= (nb << i)
    pcg32rand([st])
    return st 