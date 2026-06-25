import numpy as np
import math
import matplotlib.pyplot as plt


maxCars = 20
states = [(i, j) for i in range (0, 21) for j in range(0, 21)]
V = {s: 0.0 for s in states}
#start with 0 for every state for policy
policy = {s: 0 for s in states}

#cars requester and returned each day are poisson random variables

def actions(s):
    #example [14, 7]
    #14 at location 1, 7 at location 2, so we can move 13 to location 2 and 6 to location 1, range -13, 6

    #20 - 14, 20 - 7, min(maxCars - s[0], s[1]), min(maxCars - s[1], s[0])
    #-6, 13, -6 means move 6 left, 13 means move 13 right

    return [i for i in range(-min(5, min(maxCars - s[0], s[1])), min(5, min(maxCars - s[1], s[0])) + 1)]

def poisson(expected, n):
    val = (pow(expected, n) / math.factorial(n)) * ( 1 / math.exp(expected))
    return val

def q(s, a, lr):
    value = 0
    for rent1 in range(0, 11):
                rent1prob = poisson(3, rent1)
                for rent2 in range(0, 11):
                    rent2prob = poisson(4, rent2)
                    for return1 in range(0, 11):
                        return1prob = poisson(3, return1)
                        for return2 in range(0, 11):
                            return2prob = poisson(2, return2)

                            conditionalProb = rent1prob * rent2prob * return1prob * return2prob
                            avail1 = max(s[0] - a, 0)
                            avail2 = max(s[1] + a, 0)
                            actualRent1 = min(rent1, avail1)
                            actualRent2 = min(rent2, avail2)
                            reward = abs(a) * -2 + (actualRent1 + actualRent2) * 10

                            ##for each next states we also rent and return, but we can have min and max, 0, 20
                            nextState = (min(max(avail1 - actualRent1 + return1, 0), maxCars),
             min(max(avail2 - actualRent2 + return2, 0), maxCars))
                            value += conditionalProb * (reward + lr * V[nextState])
    return value

def policyEval(threshold, lr):
    while(1):
        delta = 0
        for s in states:
            v = V[s]
            #loop through, s', r, reward is deterministic, but s' isn't
            #return and rented for each location is random, so loop thru
            V[s] = q(s, policy[s], lr)
            delta = max(delta, abs(v - V[s]))
        
        if delta < threshold:
            break


def policyImprovement(lr):
    policyStable = True

    for s in states:
        oldAction = policy[s]
        actionList = actions(s)
        qs = [q(s, a, lr) for a in actionList]
        best = actionList[qs.index(max(qs))]
        policy[s] = best
        if best != oldAction:
            policyStable = False
    return policyStable

def main():
    while (1):
        policyEval(1e-4, 0.9)
        if policyImprovement(0.9):
             break
        grid = np.array([[policy[(i,j)] for j in range(21)] for i in range(21)])
        plt.imshow(grid, origin='lower', cmap='coolwarm')
        plt.colorbar(label='cars moved')
        plt.xlabel('cars at loc 2'); plt.ylabel('cars at loc 1')
        plt.show()
    

main()
grid = np.array([[policy[(i,j)] for j in range(21)] for i in range(21)])
plt.imshow(grid, origin='lower', cmap='coolwarm')
plt.colorbar(label='cars moved')
plt.xlabel('cars at loc 2'); plt.ylabel('cars at loc 1')
plt.show()
