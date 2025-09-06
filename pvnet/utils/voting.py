import numpy as np
import random

def ransac_voting(vector_field, mask, num_hypotheses=64, threshold=0.99):
    H, W = mask.shape
    keypoints = []

    for k in range(vector_field.shape[0] // 2):
        vec_k = vector_field[2*k:2*k+2, :, :]
        votes = []

        ys, xs = np.where(mask)
        for _ in range(num_hypotheses):
            i1, i2 = random.sample(range(len(xs)), 2)
            p1 = np.array([xs[i1], ys[i1]])
            p2 = np.array([xs[i2], ys[i2]])
            d1 = vec_k[:, ys[i1], xs[i1]]
            d2 = vec_k[:, ys[i2], xs[i2]]

            A = np.array([d1, -d2]).T
            b = p2 - p1
            if np.linalg.matrix_rank(A) < 2:
                continue

            t = np.linalg.lstsq(A, b, rcond=None)[0]
            vote_point = p1 + t[0]*d1
            votes.append(vote_point)

        votes = np.array(votes)
        mean = votes.mean(axis=0)
        cov = np.cov(votes.T)
        keypoints.append((mean, cov))

    return keypoints
