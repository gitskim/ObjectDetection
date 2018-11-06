try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q


def read_from_file(filename, probability, amnt):
    r = open(filename, "r")
    info = []
    for line in r:
        if line[0] != "[":
            line = line.replace("\'", "")
            info.append(line.replace("]", "").strip())
    prob = []
    for l in info:
        temp = l.split(" ")
        for t in temp:
            if t != "\n":
                prob.append(t)

    q = Q.PriorityQueue()
    d = {}
    for p in prob:
        info = p.split(":")
        number = float(info[0])
        if number >= float(probability):
            if info[1] not in d:
                d[info[1]] = 0
            d[info[1]] += 1

    for key, val in d.items():
        q.put((-val, key))

    ans = []
    for i in range(amnt):
        if not q.empty():
            ans.append(q.get())

    print(ans)


read_from_file("dont_commit.txt", 0.1, 5)
