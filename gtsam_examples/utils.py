import matplotlib.pyplot as plt
from gtsam import symbolChr, symbolIndex
import numpy as np

def plot_chain(values, keys_to_plot, connections, title="Kinematic Chain", arrow_len=1.0, ax=None):
    """Plot selected poses as (x,y) points with orientation arrows, connected by lines."""
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(7, 7))
    idx = {}
    xs, ys, thetas, labels = [], [], [], []
    for i, key in enumerate(keys_to_plot):
        p = values.atPose2(key)
        xs.append(p.x())
        ys.append(p.y())
        thetas.append(p.theta())
        labels.append(f"{chr(symbolChr(key))}{symbolIndex(key)}")
        idx[key] = i
    for k1, k2 in connections:
        if k1 in idx and k2 in idx:
            i1, i2 = idx[k1], idx[k2]
            ax.plot([xs[i1], xs[i2]], [ys[i1], ys[i2]], 'b-', alpha=0.4, lw=2)
    for x, y, th in zip(xs, ys, thetas):
        dx = arrow_len * np.cos(th)
        dy = arrow_len * np.sin(th)
        ax.arrow(x, y, dx, dy, head_width=0.3, head_length=0.3, fc='r', ec='r', alpha=0.7)
    ax.scatter(xs, ys, s=60, c='blue', zorder=3)
    for lab, x, y in zip(labels, xs, ys):
        ax.annotate(lab, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)
    if own_fig:
        plt.show()

def get_connections(graph):
    """Extract kinematic connections from graph factors (skipping priors/1-key factors)."""
    connections = []
    for i in range(graph.size()):
        factor = graph.at(i)
        keys = list(factor.keys())
        if len(keys) >= 2:
            for j in range(len(keys) - 1):
                connections.append((keys[j], keys[j+1]))
    return connections

def plot_side_by_side(values1, values2, keys_to_plot, connections, title1="Initial Estimate", title2="Optimized Result", arrow_len=1.0):
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_chain(values1, keys_to_plot, connections, "Initial Estimate", ax=axes[0])
    plot_chain(values2, keys_to_plot, connections, "Optimized Result", ax=axes[1])
    plt.tight_layout()
    plt.show()