import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    T = int(data[ptr])
    ptr += 1
    output = []
    for _ in range(T):
        n, m = int(data[ptr]), int(data[ptr+1])
        ptr += 2
        a = list(map(int, data[ptr:ptr+n]))
        ptr += n
        
        sorted_pairs = sorted((a[i], i) for i in range(n))
        to_delete = [False] * n
        for i in range(m):
            to_delete[sorted_pairs[i][1]] = True
        
        res = []
        for i in range(n):
            if not to_delete[i]:
                res.append(str(a[i]))
        output.append(' '.join(res))
    
    print('\n'.join(output))

if __name__ == "__main__":
    main()