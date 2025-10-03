import argparse
from .pcg32 import pcg32rand, pcg32seed, mathrand
from .bruteforce import bruteseq

def main():
    p = argparse.ArgumentParser(description='clover/CLOVER - gpu/cpu bruteforce tool for math.random PCG32 sequences')
    p.add_argument('-brute', '--brute', type=str, help='bruteforce from list of comma seperated/space seperated ints.')
    p.add_argument('-stategrab', '--stategrab', type=int, help='generate the state for a given seed')
    p.add_argument('-next', '--next', type=int, help='generate states sequentially from one state')
    p.add_argument('-cpu', '--usecpu', action='store_true', help='force CPU mode (disable GPU)')
    args = p.parse_args()

    if args.brute:
        seq = [int(x) for x in args.brute.replace(',', ' ').split()] # yeah
        st = bruteseq(seq, gpuburn=not args.cpu)
        print('found states:', st)
    elif args.stategrab is not None:
        st = [0]
        pcg32seed(st, args.stategrab)
        print('state value:', st[0])
    elif args.next is not None:
        st = [args.next]
        print('seq for state', st[0])
        for _ in range(10):
            pcg32rand(st)
            print(st[0])
    else:
        p.print_help()

if __name__ == '__main__':
    main()