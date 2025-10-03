import numpy as np
from .pcg32 import pcg32rand, unxorsh, PCG32_CONST, PCG32_INC
try:
    from .brutegpu import numabruter
    hasgpu = True
except ImportError:
    hasgpu = False
import concurrent.futures
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _cpu_worker(args):
    out, r, lstart, lend, seq, chunk_size = args  
    found_states = []
    xsh = ((out << r) | (out >> ((-r) & 31))) & 0xFFFFFFFF
    highbits = xsh << 27
    logging.info(f"scanning range {lstart}-{lend} with rotation: {r}")
    for low in range(lstart, lend):
        if low % (chunk_size // 5) == 0: 
            logging.info(f"scanning: rotation={r}, progress={low-lstart}/{lend-lstart}")
        fullxor = (highbits ^ low) | (r << 59)
        st = unxorsh(fullxor)
        st_arr = [st]
        logging.debug(f"attempt low={low}, fullxor={fullxor}, state={st}")
        ok = True
        for s in seq:
            if pcg32rand(st_arr) != s:
                ok = False
                break
        if ok:
            logging.info(f"found good state: {st}")
            found_states.append(st)
    return found_states

def bruteseq(seq, gpuburn=True):
    if gpuburn and hasgpu:
        return numabruter(seq, verbose=True)
    else:
        out = seq[0]
        n = 1 << 27
        num_workers = os.cpu_count() or 4
        logging.info(f"[CLOVER] beginning cpu burn with {num_workers} workers...")
        chunk_size = n // num_workers // 2  
        tasks = []
        for r in range(32):
            for i in range(num_workers):
                lstart = i * chunk_size
                lend = (i + 1) * chunk_size if i < num_workers - 1 else n
                tasks.append((out, r, lstart, lend, seq, chunk_size))
                logging.info(f"[CLOVER] deb rotation {r}, range {lstart}-{lend}")
        found_states = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for result in executor.map(_cpu_worker, tasks):
                found_states.extend(result)
                logging.info(f"[CLOVER] cpu worker found {len(result)} states.")
        return found_states