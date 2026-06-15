import numpy as np
import matplotlib.pyplot as plt

def simpleBanditAlgo(Q, N, q, epsilon, run):
    rewards = []
    optimals = []
    rng = np.random.default_rng(seed=run)
    for step in range(10000):
        prob = rng.random()
        action_index = -1
        if prob > epsilon:
            action_index = rng.choice(np.flatnonzero(Q == Q.max()))
        else:
            action_index = rng.integers(0, 10)
        
        R = rng.normal(q[action_index], 1)
        optimal_step = (action_index == np.argmax(q))
        N[action_index] = N[action_index] + 1
        Q[action_index] = Q[action_index] + (1 / N[action_index]) * (R - Q[action_index])

        q += rng.standard_normal(10) * 0.01
        rewards.append(R)
        optimals.append(optimal_step)
    return rewards, optimals

def constantBanditAlgo(Q, q, epsilon, a, run):
    rewards = []
    optimals = []
    rng = np.random.default_rng(seed=run)
    for step in range(10000):
        prob = rng.random()
        action_index = -1
        if prob > epsilon:
            action_index = rng.choice(np.flatnonzero(Q == Q.max()))
        else:
            action_index = rng.integers(0, 10)
        
        R = rng.normal(q[action_index], 1)
        optimal_step = (action_index == np.argmax(q))
        Q[action_index] = Q[action_index] + a * (R - Q[action_index])

        q += rng.standard_normal(10) * 0.01
        rewards.append(R)
        optimals.append(optimal_step)
    return rewards, optimals
        
def main():
    epsilon = 0.1

    sa_rewards = np.zeros(10000)
    sa_optimals = np.zeros(10000)
    for i in range(200):
        q = np.ones(10)
        Q = np.zeros(10)
        N = np.zeros(10)
        r, o = simpleBanditAlgo(Q, N, q, epsilon, i + 1)
        sa_rewards += r
        sa_optimals += o
    sa_rewards /= 200
    sa_optimals /= 200

    cs_rewards = np.zeros(10000)
    cs_optimals = np.zeros(10000)
    for i in range(200):
        q = np.ones(10)
        Q = np.zeros(10)
        r, o = constantBanditAlgo(Q, q, epsilon, 0.1, i + 1)
        cs_rewards += r
        cs_optimals += o
    cs_rewards /= 200
    cs_optimals /= 200

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

    ax1.plot(sa_rewards, label="sample average (1/n)")
    ax1.plot(cs_rewards, label="constant step-size (alpha=0.1)")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average reward")
    ax1.legend()

    ax2.plot(sa_optimals * 100, label="sample average (1/n)")
    ax2.plot(cs_optimals * 100, label="constant step-size (alpha=0.1)")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% optimal action")
    ax2.set_ylim(0, 100)
    ax2.legend()

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
