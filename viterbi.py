import sys
import numpy as np

def viterbi_forward(data, m_r, m_c, observations_raw):

    map_data = [row.replace(' ', '') for row in data[1:m_r+1]] 

    # Calculate K and N
    grid = np.ones((m_r, m_c))
    K = 0
    for i in range(m_r):
        for j in range(m_c):
            if map_data[i][j] == 'X':
                grid[i, j] = np.inf
            else:
                grid[i, j] = 0
                K = K + 1

    err_rate = float(data[-1])
    N = observations_raw

    # Observations
    observations = data[m_r + 2 : m_r + 2 + observations_raw]

    Y = []
    for line in observations:
        observation = int(line, 2)
        Y.append(observation)

    S = []
    for i in range(m_r):
        for j in range(m_c):
            if grid[i, j] == 0:
                S.append((i, j))

    Q = np.full(K, 1/K)

    Tm = np.zeros((K, K))
    for i in range(K):
        neighbors = []
        for n_x, n_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = S[i]
            x = x + n_x
            y = y + n_y
            if 0 <= x < m_r and 0 <= y < m_c and grid[x, y] == 0:
                neighbors.append(S.index((x, y)))

        for j in neighbors:
            Tm[i, j] = 1 / len(neighbors)

    Em = np.zeros((K, 2**4))
    for i in range(K):
        x, y = S[i]

        for o in range(2**4):
            bit_hack = format(o, '04b')
            error = 0
            for x, y, bit in zip([-1, 0, 1, 0], [0, 1, 0, -1], bit_hack):
                n_x = x + x
                n_y = y + y
                if (0 <= x < m_r and 0 <= y < m_c):
                    if (grid[x, y] == 1) != (bit == '1'):
                        error += 1
                else:
                    if bit == '1':
                        error += 1

            Em[i, o] = ((1 - err_rate)**(4 - error)) * (err_rate**error)
            
    trellis = np.zeros((K, N))
    for i in range(K):
        trellis[i, 0] = Q[i] * Em[i, Y[0]]

    for j in range(1, N):
        for i in range(K):
            trellis[i, j] = max(trellis[k, j - 1] * Tm[k, i] * Em[i, Y[j]] for k in range(K))

    maps = []
    for j in range(N):
        map_prob = np.zeros((m_r, m_c))
        for i in range(K):
            x, y = S[i]
            map_prob[x, y] = trellis[i, j]
        maps.append(map_prob)

    np.savez("output.npz", *maps)
    data = np.load('output.npz')
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])


with open(sys.argv[1], 'r') as raw_data:
    rows = raw_data.read().strip().split("\n")
    map_size = rows[0].split()
    sensor_ob = int(rows[int(map_size[0]) + 1])
    viterbi_forward(rows, int(map_size[0]), int(map_size[1]), sensor_ob)






