import copy
from collections import deque

class Bfs:
    dr = [0, 0, 0, -1, 1]
    dc = [0, -1, 1, 0, 0]

    def in_corner(self, p):
        if p[0] == 1 or p[1] == 1:
            return False
        return True

    def check_done(self, ar):
        for unit in range(3):
            if not self.in_corner(ar[unit]):
                return False
        return True

    def check_valid(self, ar):
        for unit in range(3):
            for i in range(2):
                if not (0 <= ar[unit][i] <= 2):
                    return False
        if len(set(str(x) for x in ar)) != 3:
            return False
        return True

    def solve(self, ar):
        q = deque()
        q.append((ar, [ar]))

        vis = set()

        while len(q) > 0:
            ar, hist = q.popleft()
            vis.add(str(ar))
            ars = set(str(x) for x in ar)
            if self.check_done(ar):
                return hist

            for k1 in range(5):
                for k2 in range(5):
                    for k3 in range(5):
                        tar = copy.deepcopy(ar)
                        temp_hist = copy.deepcopy(hist)
                        if not self.in_corner(tar[0]):
                            tar[0][0] += self.dr[k1]
                            tar[0][1] += self.dc[k1]
                            if str(tar[0]) in ars and k1 != 0:
                                continue
                        if not self.in_corner(tar[1]):
                            tar[1][0] += self.dr[k2]
                            tar[1][1] += self.dc[k2]
                            if str(tar[1]) in ars and k2 != 0:
                                continue
                        if not self.in_corner(tar[2]):
                            tar[2][0] += self.dr[k3]
                            tar[2][1] += self.dc[k3]
                            if str(tar[2]) in ars and k3 != 0:
                                continue
                        if self.check_valid(tar) and not (str(tar) in vis):
                            temp_hist.append(tar)
                            q.append((tar, temp_hist))

bfs = Bfs()
#print(bfs.solve([[0, 0], [0, 1], [2, 1]]))

for k1 in range(2):
    for k2 in range(2):
        for k3 in range(2):
            for k4 in range(2):
                for k5 in range(2):
                    for k6 in range(2):
                        ar = [[k1, k2],
                              [k3, k4],
                              [k5, k6]]
                        if not bfs.check_valid(ar):
                            continue
                        print(ar)
                        print(bfs.solve(ar))
                        print()
