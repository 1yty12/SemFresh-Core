"""Optional actor-loss composition; caller provides its own value network."""
import tensorflow as tf

def core_actor_loss(model, state, action, lagrange, value_fn, *,
                    pds_weight, guidance_weight, cost_scale, **context):
    if cost_scale <= 0:
        raise ValueError('cost_scale must be positive')
    post = model.build_pds(state, action, **context)
    critic_term = -tf.reduce_mean(value_fn(post))
    pds_term = tf.reduce_mean(model.estimate_cost(state, action, lagrange, **context)) / cost_scale
    target = tf.stop_gradient(model.get_math_optimal_action(state, action[:, :, 2:]))
    guidance_term = tf.reduce_mean(tf.square(action[:, :, :2] - target[:, :, :2]))
    total = critic_term + pds_weight * pds_term + guidance_weight * guidance_term
    return total, dict(critic=critic_term, pds=pds_term, guidance=guidance_term)
