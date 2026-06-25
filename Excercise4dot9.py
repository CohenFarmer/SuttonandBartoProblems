states = [i for i in range(0, 101)]
V = {s: 0.0 for s in states}

def action(s):
    return [i for i in range(1, min(s + 1, 100 - s + 1))]

def q(s, a, probHeads, lr):
    r = 0
    leftover = s - a
    vHeads = 0
    vTails = V[leftover]
    if leftover + a * 2 == 100:
        r = 1
    
    if leftover + a * 2 > 100:
        vHeads = 0
    else:
        vHeads = V[leftover + a * 2]

    heads = probHeads * (r + lr * vHeads)
    tails = (1 - probHeads) * (lr * vTails)
    return heads + tails


def valueIteration(probHeads, threshold, lr):
    while (1):
        delta = 0
        for s in states:
            if s == 0 or s == 100:
                continue
            v = V[s]
            maxA = 0
            for a in action(s):
                maxA = max(maxA, q(s, a, probHeads, lr))
            #get best action value for now
            V[s] = maxA
            delta = max(delta, abs(v - V[s]))
        if delta <= threshold:
            break
    policy = {s: 0 for s in states}
    for s in states:
        maxA = 0
        maxAction = 0
        for a in action(s):
            qVal = q(s, a, probHeads, lr)
            if qVal > maxA:
                maxA = qVal
                maxAction = a
        policy[s] = maxAction
    return policy
        
def main():
    policy = valueIteration(0.51, 1e-4, 1)
    print(policy)

main()

