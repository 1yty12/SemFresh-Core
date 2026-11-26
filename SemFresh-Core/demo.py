"""Synthetic mechanism/gradient check only; not a paper experiment."""
import numpy as np
import tensorflow as tf
from semantic_core import CoreConfig, PDSModel, core_actor_loss


def main():
    # Deliberately tiny dimensionless toy inputs, not experimental parameters.
    config = CoreConfig(N=2, M=2, f_max=[2., 3.], P_max=[1., 1.],
                        d_t=1., kappa=1., sigma2=1., W_max=4., gamma=.02,
                        d_ub=1., lambda_adoi=.3, F_edge_max=6., edge_load_alpha=.2)
    model = PDSModel(config, e_max=[.4, .4])
    state = np.zeros((1, 2, 9), np.float32)
    state[:, :, 0] = [1e10, 2e10]
    state[:, :, 1:7] = [.3, .4, .7, .2, .6, .4]
    state[:, :, -2] = np.log1p(.5)
    state[:, :, -1] = np.log1p(.7*(1-np.exp(-.3*.4))+.4*(1-np.exp(-.3*.6)))
    action = tf.Variable([[[0., 0., -.7, .7], [0., 0., -.6, .6]]], dtype=tf.float32)
    parents = np.array([[[0, 1], [0, 1]]], np.int32)
    priorities = np.ones((1, 2, 2), np.int32)
    importances = state[:, :, [3, 6]]
    context = dict(parent_ids=parents, priorities=priorities, importances=importances)
    with tf.GradientTape() as tape:
        loss, parts = core_actor_loss(model, state, action, tf.ones(2),
            lambda post: -tf.reduce_sum(post[:, :, -1], axis=1),
            pds_weight=.2, guidance_weight=.3, cost_scale=1., **context)
    grad = tape.gradient(loss, action)
    assert grad is not None and np.isfinite(grad.numpy()).all()
    with tf.GradientTape() as tape:
        local, offload = model.get_decision_probs(action)
        gate_sum = tf.reduce_sum(local)
    gate_grad = tape.gradient(gate_sum, action)
    np.testing.assert_array_equal(local.numpy(), [[[1., 0.], [1., 0.]]])
    np.testing.assert_array_equal(offload.numpy(), [[[0., 1.], [0., 1.]]])
    assert np.any(np.abs(gate_grad.numpy()[:, :, 2:]) > 0)
    post = model.build_pds(state, action, **context)
    exact = model.build_pds_exact_batch(state, action.numpy(), parents, priorities, importances, [0])
    target = model.get_math_optimal_action(state, action[:, :, 2:])
    assert post.shape == state.shape == exact.shape
    assert np.isfinite(post.numpy()).all() and np.isfinite(exact).all()
    assert np.isfinite(target.numpy()).all()
    assert np.max(np.abs(target.numpy())) <= 1.
    empty = state.copy()
    empty[:, :, 1:] = 0.
    assert np.isfinite(model.build_pds(empty, action).numpy()).all()
    print('PASS: hard forward gates, nonzero STE gradient, finite loss gradients, PDS shapes, exact path, empty queue and bounded guidance.')
    print('TensorFlow', tf.__version__, 'NumPy', np.__version__)
    print({key: float(value.numpy()) for key, value in parts.items()})

if __name__ == '__main__':
    main()
