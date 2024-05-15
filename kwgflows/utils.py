
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import jax
import jax.numpy as jnp
import numpy as np
np.random.seed(49)
import ot
import ott
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from densratio import densratio

from kwgflows.divergences.mmd import *

plt.rcParams['axes.grid'] = True
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 20
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()

plt.rc('font', size=20)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=18, frameon=False)
plt.rc('xtick', labelsize=14, direction='in')
plt.rc('ytick', labelsize=14, direction='in')
plt.rc('figure', figsize=(6, 4))

FLOW_LIST = ['mmd', 'drmmd']

def compute_wasserstein_distance_numpy(X, Y):
    a, b = jnp.ones((X.shape[0], )) / X.shape[0], jnp.ones((Y.shape[0], )) / Y.shape[0]
    M = ot.dist(X, Y, 'euclidean')
    W = ot.emd(a, b, M)
    Wd = (W * M).sum()
    return Wd

@jax.jit
def compute_wasserstein_distance_jax(X, Y):
    """
    This is the jax implementation for computing the Wasserstein distance.
    However, it can only optimize the sinkorn divergence.
    When setting the epsilon=0.0, the optimization returns zero.
    Not sure why.
    """
    cost_fn = costs.PNormP(p=1)
    geom = pointcloud.PointCloud(X[:, None], Y[:, None], cost_fn=cost_fn, epsilon=0.001)
    ot_prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    ot = solver(ot_prob)
    Wd = jnp.sum(ot.matrix * ot.geom.cost_matrix)
    return Wd


def compute_wasserstein_distance_trajectory(flow_1, flow_2, eval_freq):
    assert flow_1.shape[0] == flow_2.shape[0]
    T = flow_1.shape[0]
    wasserstein_distance = []
    for i in range(0, T, eval_freq):
        wasserstein_distance.append(compute_wasserstein_distance_numpy(flow_1[i, :], flow_2[i, :]))
    wasserstein_distance = jnp.array(wasserstein_distance)
    return wasserstein_distance

def evaluate(args, ret, rate):
    # Save the trajectory
    eval_freq = rate
    jnp.save(f'{args.save_path}/Ys.npy', ret.Ys[::eval_freq, :])

    T = ret.Ys.shape[0]
    X = ret.divergence.X

    wass_distance = compute_wasserstein_distance_trajectory(ret.Ys, jnp.repeat(X[None, :], T, axis=0), eval_freq)
    
    mmd_divergence = mmd_fixed_target(args, args.kernel_fn, None)
    mmd_divergence.pre_compute(X)
    mmd_distance = jnp.sqrt(jax.vmap(mmd_divergence)(ret.Ys[::eval_freq, :]))

    drmmd_divergence = drmmd_fixed_target(args, args.kernel_fn, None)
    drmmd_divergence.pre_compute(X)
    drmmd_distance = jax.vmap(drmmd_divergence)(ret.Ys[::eval_freq, :])

    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].plot(wass_distance, label='Wass 2')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Wasserstein 2 distance')
    axs[1].plot(mmd_distance, label='mmd')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('MMD distance')
    axs[2].plot(drmmd_distance, label='drmmd')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('drmmd distance')
    plt.savefig(f'{args.save_path}/distance.png')
    return 


def save_animation_1d(args, ret, rate, save_path):
    num_timesteps = ret.Ys.shape[0]
    num_frames = max(num_timesteps // rate, 1)
    num_timesteps_grid = np.arange(1, num_timesteps+1, rate)

    # Combined update function for both animations
    def update(frame):
        # Update for density ratio
        densratio_obj = densratio(np.array(ret.get_Yt(frame * rate)), X, alpha=alpha, verbose=0)
        _animate_density_ratio.set_xdata(grid)
        _animate_density_ratio.set_ydata(densratio_obj.compute_density_ratio(grid))
        
        # Update for scatter plot
        x = jnp.clip(ret.get_Yt(frame * rate), -1, 1)
        y = ret.get_Yt(frame * rate) * 0.0
        data = np.concatenate((x, y), axis=-1)
        _animate_scatter.set_offsets(data)
        
        _animate_distance.set_xdata(num_timesteps_grid[:frame+1])
        _animate_distance.set_ydata(drmmd_distance[:frame+1])
        return (_animate_density_ratio, _animate_scatter, _animate_distance)

    alpha = 0.1
    X = np.array(ret.divergence.X)
    densratio_obj = densratio(np.array(ret.get_Yt(0)), X, alpha=alpha, verbose=0)
    grid = np.linspace(X.min(), X.max(), 100)

    # Create a single figure with two subplots
    animate_fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Initial plot for density ratio on the first subplot
    _animate_density_ratio, = ax1.plot(grid, densratio_obj.compute_density_ratio(grid), label='Density Ratio')
    ax1.set_title(r'Density ratio $\frac{\mu_t}{\pi}$')

    drmmd_divergence = drmmd_fixed_target(args, args.kernel_fn, None)
    drmmd_divergence.pre_compute(ret.divergence.X)
    drmmd_distance = jax.vmap(drmmd_divergence)(ret.Ys[::rate, :])
    drmmd_distance = np.array(drmmd_distance)
    ax2.set_title(r'$\text{DrMMD}(\mu_t \| \pi)$')
    ax2.set_xlabel('Iteration')
    ax2.set_xlim([0, num_timesteps])
    ax2.set_ylim([0.0, 1.3])
    _animate_distance, = ax2.plot(num_timesteps_grid[0], drmmd_distance[0], label='drmmd')

    ax3.scatter(ret.divergence.X[:, 0], ret.divergence.X[:, 0] * 0.0, label=r'$\pi$')
    _animate_scatter = ax3.scatter(jnp.clip(ret.get_Yt(0)[:, 0], -1, 1), ret.get_Yt(0)[:, 0] * 0.0, label=r'$\mu_t$')
    ax3.set_xlim(-1.5, 1.5)
    ax3.axis("off")
    ax3.legend()

    # Create a single FuncAnimation for both updates
    ani_combined = FuncAnimation(
        animate_fig,
        update,
        frames=num_frames,
        blit=True,
        interval=50,
    )

    ani_combined.save(f'{save_path}/animation_combined.mp4', writer='ffmpeg', fps=1)
    return

def save_animation_2d(args, ret, rate, save_path):
    num_timesteps = ret.Ys.shape[0]
    num_frames = max(num_timesteps // rate, 1)

    def update(frame):
        _animate_scatter.set_offsets(ret.get_Yt(frame * rate)[:, ::-1])
        return (_animate_scatter,)

    # create initial plot
    animate_fig, animate_ax = plt.subplots()
    # animate_fig.patch.set_alpha(0.)
    # plt.axis('off')
    # animate_ax.scatter(ret.Ys[:, 0], ret.Ys[:, 1], label='source')
    if args.dataset == 'ThreeRing':
        animate_ax.set_xlim(-2.0, 1.0)
        animate_ax.set_ylim(-1.0, 1.0)

    # awkard way to share state for now
    animate_ax.scatter(ret.divergence.X[:, 1], ret.divergence.X[:, 0], label='target')
    _animate_scatter = animate_ax.scatter(ret.get_Yt(0)[:, 1], ret.get_Yt(0)[:, 0], label='target')

    ani_kale = FuncAnimation(
        animate_fig,
        update,
        frames=num_frames,
        # init_func=init,
        blit=True,
        interval=50,
    )
    ani_kale.save(f'{save_path}/animation.mp4',
                   writer='ffmpeg', fps=20)
    return    
