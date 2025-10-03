import numpy as np
from .pcg32 import PCG32_CONST, PCG32_INC
import time
try:
    from numba import cuda
    numba_available = True
except ImportError:
    numba_available = False

if numba_available:
    @cuda.jit(device=True)
    def uxsnumba(fullxor):
        st = fullxor & (~((1 << 46) - 1))
        for j in range(63 - 18, -1, -1):
            upb = (st >> (j + 18)) & 1
            cur = (fullxor >> j) & 1
            nb = upb ^ cur
            st |= (nb << j)
        return st

    @cuda.jit
    def kernelburn(out, seq, seq_len, valid_states, valid_count, max_results):
        idx = cuda.grid(1)
        stride = cuda.gridsize(1)
        for i in range(idx, 1 << 27, stride):
            if idx == 0 and i % (1 << 20) == 0:  
                print(f"[KBURN] proglog: i={i}, idx={idx}")
            for rot in range(32):
                xsh = ((out << rot) | (out >> ((-rot) & 31))) & 0xFFFFFFFF
                highbits = xsh << 27  
                fullxor = (highbits ^ i) | (rot << 59)
                st = uxsnumba(fullxor)
                nextst = st
                if idx == 0 and rot % 8 == 0: 
                    print(f"[KBURN] idx={idx}, i={i}, rot={rot}, fullxor={fullxor}, st={st}")
                ok = True
                for s in range(seq_len):
                    oldst = nextst
                    nextst = (oldst * PCG32_CONST + (PCG32_INC | 1)) & 0xFFFFFFFFFFFFFFFF
                    xsh2 = ((oldst >> 18) ^ oldst) >> 27
                    rot2 = oldst >> 59
                    res = ((xsh2 >> rot2) | (xsh2 << ((-rot2) & 31))) & 0xFFFFFFFF
                    if seq[s] != res:
                        ok = False
                        break
                if ok:
                    pos = cuda.atomic.add(valid_count, 0, 1)
                    if pos < max_results:
                        valid_states[pos] = st
                        if idx == 0:  
                            print(f"[KERNEL] valid state: {st}, pos={pos}")

def numabruter(seq, max_results=1024, verbose=True):
    if not numba_available:
        raise ImportError('fuck off')
    out = int(seq[0])
    seq_arr = np.array(seq, dtype=np.uint32)
    valid_states = np.zeros(max_results, dtype=np.uint64)
    valid_count = np.zeros(1, dtype=np.uint32)
    d_seq = cuda.to_device(seq_arr)
    d_valid_states = cuda.to_device(valid_states)
    d_valid_count = cuda.to_device(valid_count)
    threads = 256
    blocks = 256
    if verbose:
        print(f"[CLOVER] beginning CLOVER gpu kernelburn with {blocks} blocks, {threads} threads per block")
        print(f"[CLOVER] scanning for states matching sequence: {seq}")
        t0 = time.time()
    kernelburn[blocks, threads](out, d_seq, len(seq), d_valid_states, d_valid_count, max_results)
    cuda.synchronize()
    if verbose:
        t1 = time.time()
        print(f"[CLOVER] kernel finished in {t1-t0:.2f} seconds")
    d_valid_states.copy_to_host(valid_states)
    d_valid_count.copy_to_host(valid_count)
    if verbose:
        print(f"[CLOVER] found {valid_count[0]} valid states.")
    return valid_states[:valid_count[0]]