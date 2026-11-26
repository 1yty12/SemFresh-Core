"""Core mechanisms extracted from the proposed algorithm; no training environment."""
import numpy as np
import tensorflow as tf

EPS = 1e-8

def calculate_noma_rate_np(power, channel, sigma2, bandwidth):
    power = np.asarray(power, dtype=np.float32).reshape(-1)
    channel = np.asarray(channel, dtype=np.float32).reshape(-1)
    sort_indices = np.argsort(channel)[::-1]
    power_sorted = power[sort_indices]
    channel_sorted = channel[sort_indices]
    rx_power = power_sorted * channel_sorted
    total_power = np.cumsum(rx_power[::-1])[::-1]
    interference = total_power - rx_power
    sinr = rx_power / (sigma2 + interference + 1e-12)
    rates_sorted = bandwidth * np.log2(1.0 + sinr)
    unsort_indices = np.argsort(sort_indices)
    return rates_sorted[unsort_indices].astype(np.float32)

def calculate_noma_rate_tf(power, channel, sigma2, bandwidth):
    rx_power = power * channel
    channel_col = tf.expand_dims(channel, axis=2)
    channel_row = tf.expand_dims(channel, axis=1)
    interference_mask = tf.cast(channel_row < channel_col, dtype=tf.float32)
    rx_power_row = tf.expand_dims(rx_power, axis=1)
    interference = tf.reduce_sum(rx_power_row * interference_mask, axis=2)
    sinr = rx_power / (sigma2 + interference + 1e-12)
    return bandwidth * tf.math.log(1.0 + sinr) / tf.math.log(2.0)

class PDSModel:

    def __init__(self, config, e_max, workload_trace=None, gate_type='Hard_STE'):
        self.param = config
        self.task_feature_dim = 3
        self.gate_type = gate_type
        if gate_type not in ('Hard_STE', 'Hard', 'Soft'):
            raise ValueError('gate_type must be Hard_STE, Hard, or Soft')
        self.e_max = np.asarray(e_max, dtype=np.float32)
        if self.e_max.shape != (config.N,) or not np.isfinite(self.e_max).all() or (self.e_max < 0).any():
            raise ValueError('e_max must contain N finite nonnegative per-slot budgets')
        self.workload_trace = np.asarray([0.0] if workload_trace is None else workload_trace, dtype=np.float32)
        if self.workload_trace.ndim != 1 or not self.workload_trace.size or (not np.isfinite(self.workload_trace).all()) or (self.workload_trace < 0).any() or (self.workload_trace > 1).any():
            raise ValueError('workload_trace must be a nonempty vector in [0, 1]')
        self.trace_len = len(self.workload_trace)
        self.f_max = tf.convert_to_tensor(self.param.f_max, dtype=tf.float32)
        self.p_max = tf.convert_to_tensor(self.param.P_max, dtype=tf.float32)
        self.d_t = tf.convert_to_tensor(self.param.d_t, dtype=tf.float32)
        self.kappa = tf.convert_to_tensor(self.param.kappa, dtype=tf.float32)
        self.sigma2 = tf.convert_to_tensor(self.param.sigma2, dtype=tf.float32)
        self.bandwidth = tf.convert_to_tensor(self.param.W_max, dtype=tf.float32)
        self.gamma_energy = tf.convert_to_tensor(self.param.gamma, dtype=tf.float32)
        self.d_ub = tf.convert_to_tensor(self.param.d_ub, dtype=tf.float32)
        self.lambda_adoi = tf.convert_to_tensor(self.param.lambda_adoi, dtype=tf.float32)
        self.f_edge_max = tf.convert_to_tensor(self.param.F_edge_max, dtype=tf.float32)
        self.edge_load_alpha = tf.convert_to_tensor(self.param.edge_load_alpha, dtype=tf.float32)

    def _get_trace_load_tf(self, trace_idx, batch_size):
        if trace_idx is None:
            return tf.zeros((batch_size,), dtype=tf.float32)
        trace_idx = tf.cast(tf.reshape(trace_idx, (-1,)), tf.int32)
        trace_idx = tf.math.floormod(trace_idx, self.trace_len)
        return tf.gather(tf.convert_to_tensor(self.workload_trace, dtype=tf.float32), trace_idx)

    def _infer_hidden_slots(self, hidden_load, hidden_penalty, num_slots):
        pseudo_tasks = []
        if hidden_load <= 1e-08 or num_slots <= 0:
            return pseudo_tasks
        total_weight = max(hidden_load / float(self.param.d_ub), 1e-08)
        avg_decay = np.clip(hidden_penalty / total_weight, 0.0, 1.0 - 1e-06)
        if avg_decay <= 0.0:
            pseudo_age = 0.0
            total_importance = 0.0
        else:
            pseudo_age = -np.log(max(1e-06, 1.0 - avg_decay)) / float(self.param.lambda_adoi)
            decay = 1.0 - np.exp(-float(self.param.lambda_adoi) * pseudo_age)
            total_importance = hidden_penalty / max(decay, 1e-08)
        remaining_hidden = hidden_load
        slots_left = num_slots
        while remaining_hidden > 1e-08 and slots_left > 0:
            chunk = min(float(self.param.d_ub), remaining_hidden / slots_left)
            size_fraction = chunk / max(hidden_load, 1e-08)
            pseudo_tasks.append((chunk / float(self.param.d_ub), pseudo_age, total_importance * size_fraction))
            remaining_hidden -= chunk
            slots_left -= 1
        return pseudo_tasks

    def get_math_optimal_action(self, batch_state, batch_x):
        batch_state = tf.cast(batch_state, tf.float32)
        batch_x = tf.cast(batch_x, tf.float32)
        is_local = tf.cast(batch_x < -0.33, tf.float32)
        is_offload = tf.cast(batch_x > 0.33, tf.float32)
        h_real = batch_state[:, :, 0] / 10000000000.0
        task_sizes_norm = tf.stack([batch_state[:, :, 1 + self.task_feature_dim * k] for k in range(self.param.M)], axis=-1)
        task_sizes = task_sizes_norm * self.d_ub
        local_bits = tf.reduce_sum(task_sizes * is_local, axis=-1)
        f_req = local_bits * self.kappa / self.d_t
        f_real = tf.clip_by_value(f_req, 0.0, self.f_max)
        offload_bits = tf.reduce_sum(task_sizes * is_offload, axis=-1)
        rate_req = offload_bits / self.d_t
        gamma_req = tf.maximum(2.0 ** (rate_req / tf.maximum(self.bandwidth, 1.0)) - 1.0, 0.0)
        gamma_mat = tf.linalg.diag(gamma_req)
        h_i = tf.expand_dims(h_real, axis=2)
        h_j = tf.expand_dims(h_real, axis=1)
        mask = tf.cast(h_i < h_j, tf.float32)
        g_mat = mask * (h_j / (h_i + 1e-12))
        v_vec = gamma_req * self.sigma2 / (h_real + 1e-12)
        v_vec = tf.expand_dims(v_vec, axis=-1)
        batch_size = tf.shape(h_real)[0]
        identity = tf.eye(self.param.N, batch_shape=[batch_size])
        a_mat = identity - tf.matmul(gamma_mat, g_mat)
        p_req = tf.squeeze(tf.matmul(tf.linalg.pinv(a_mat), v_vec), axis=-1)
        p_real = tf.clip_by_value(p_req, 0.0, self.p_max)
        f_norm = tf.expand_dims(2.0 * tf.math.divide_no_nan(f_real, self.f_max) - 1.0, axis=-1)
        p_norm = tf.expand_dims(2.0 * tf.math.divide_no_nan(p_real, self.p_max) - 1.0, axis=-1)
        return tf.concat([f_norm, p_norm, batch_x], axis=-1)

    def get_decision_probs(self, batch_env_action):
        act_x = batch_env_action[:, :, 2:]
        if self.gate_type == 'Hard':
            prob_loc = tf.cast(act_x < -0.33, dtype=tf.float32)
            prob_off = tf.cast(act_x > 0.33, dtype=tf.float32)
        elif self.gate_type == 'Hard_STE':
            soft_loc = tf.nn.sigmoid(-5.0 * (act_x + 0.33))
            soft_off = tf.nn.sigmoid(5.0 * (act_x - 0.33))
            hard_loc = tf.cast(act_x < -0.33, dtype=tf.float32)
            hard_off = tf.cast(act_x > 0.33, dtype=tf.float32)
            prob_loc = tf.stop_gradient(hard_loc - soft_loc) + soft_loc
            prob_off = tf.stop_gradient(hard_off - soft_off) + soft_off
        else:
            prob_loc = tf.nn.sigmoid(-5.0 * (act_x + 0.33))
            prob_off = tf.nn.sigmoid(5.0 * (act_x - 0.33))
        return (prob_loc, prob_off)

    def _sort_window_by_parent_tf(self, task_sizes, task_ages, prob_loc, prob_off, parent_ids, priorities, importances):
        if parent_ids is None:
            return (task_sizes, task_ages, prob_loc, prob_off, importances, None)
        parent_ids = tf.cast(parent_ids, tf.int32)
        priorities = tf.cast(priorities, tf.int32)
        valid_mask = task_sizes > 0.0
        large_parent = tf.fill(tf.shape(parent_ids), tf.constant(10 ** 6, dtype=tf.int32))
        large_priority = tf.fill(tf.shape(priorities), tf.constant(10 ** 6, dtype=tf.int32))
        safe_parent = tf.where(valid_mask, parent_ids, large_parent)
        safe_priority = tf.where(valid_mask, priorities, large_priority)
        sort_key = tf.cast(safe_parent, tf.float32) * 1000000000.0 + tf.cast(safe_priority, tf.float32) * 1000.0 - task_sizes
        sort_indices = tf.argsort(sort_key, axis=-1, direction='ASCENDING', stable=True)
        task_sizes = tf.gather(task_sizes, sort_indices, axis=2, batch_dims=2)
        task_ages = tf.gather(task_ages, sort_indices, axis=2, batch_dims=2)
        prob_loc = tf.gather(prob_loc, sort_indices, axis=2, batch_dims=2)
        prob_off = tf.gather(prob_off, sort_indices, axis=2, batch_dims=2)
        importances = tf.gather(importances, sort_indices, axis=2, batch_dims=2)
        sorted_parent_ids = tf.gather(parent_ids, sort_indices, axis=2, batch_dims=2)
        return (task_sizes, task_ages, prob_loc, prob_off, importances, (sort_indices, sorted_parent_ids))

    def build_pds(self, batch_state, batch_env_action, trace_idx=None, parent_ids=None, priorities=None, importances=None):
        batch_state = tf.cast(batch_state, tf.float32)
        batch_env_action = tf.cast(batch_env_action, tf.float32)
        f_ratio = (tf.clip_by_value(batch_env_action[:, :, 0], -1.0, 1.0) + 1.0) / 2.0
        p_ratio = (tf.clip_by_value(batch_env_action[:, :, 1], -1.0, 1.0) + 1.0) / 2.0
        f = f_ratio * self.f_max
        power = p_ratio * self.p_max
        channel = batch_state[:, :, 0] / 10000000000.0
        rates = calculate_noma_rate_tf(power, channel, self.sigma2, self.bandwidth)
        c_loc = f * self.d_t / self.kappa
        c_trans = rates * self.d_t
        trace_load = self._get_trace_load_tf(trace_idx, tf.shape(batch_state)[0])
        available_f_edge = self.f_edge_max * (1.0 - self.edge_load_alpha * trace_load[:, None])
        task_sizes = tf.stack([batch_state[:, :, 1 + self.task_feature_dim * k] for k in range(self.param.M)], axis=-1) * self.d_ub
        task_ages = tf.stack([batch_state[:, :, 2 + self.task_feature_dim * k] for k in range(self.param.M)], axis=-1)
        task_importances = tf.stack([batch_state[:, :, 3 + self.task_feature_dim * k] for k in range(self.param.M)], axis=-1)
        if importances is None:
            importances = task_importances
        else:
            importances = tf.cast(importances, tf.float32)
        if priorities is None:
            priorities = tf.tile(tf.reshape(tf.range(1, self.param.M + 1, dtype=tf.int32), (1, 1, self.param.M)), (tf.shape(batch_state)[0], self.param.N, 1))
        prob_loc, prob_off = self.get_decision_probs(batch_env_action)
        orig_task_ages = task_ages
        orig_importances = importances
        task_sizes, task_ages, prob_loc, prob_off, importances, sort_meta = self._sort_window_by_parent_tf(task_sizes, task_ages, prob_loc, prob_off, parent_ids, priorities, importances)
        offload_demand = tf.reduce_sum(task_sizes * prob_off, axis=-1)
        edge_budget_bits = available_f_edge * self.d_t / self.kappa
        edge_share = tf.math.divide_no_nan(edge_budget_bits * offload_demand, tf.reduce_sum(offload_demand, axis=1, keepdims=True))
        c_offload = tf.minimum(c_trans, edge_share)
        total_processed_load = tf.zeros_like(c_loc)
        total_penalty_reduction = tf.zeros_like(c_loc)
        rem_c_loc = c_loc
        rem_c_trans = c_offload
        parent_gate = tf.ones_like(c_loc)
        group_success = tf.ones_like(c_loc)
        s_post_sorted = []
        imp_post_sorted = []
        for idx in range(self.param.M):
            curr_size = task_sizes[:, :, idx]
            curr_age = task_ages[:, :, idx]
            curr_prob_loc = prob_loc[:, :, idx]
            curr_prob_off = prob_off[:, :, idx]
            curr_importance = importances[:, :, idx]
            soft_can_loc = tf.nn.sigmoid(8.0 * (rem_c_loc - curr_size))
            soft_can_off = tf.nn.sigmoid(8.0 * (rem_c_trans - curr_size))
            delta_loc = curr_size * curr_prob_loc * soft_can_loc * parent_gate
            delta_trans = curr_size * curr_prob_off * soft_can_off * parent_gate
            processed = delta_loc + delta_trans
            rem_c_loc = tf.nn.relu(rem_c_loc - delta_loc)
            rem_c_trans = tf.nn.relu(rem_c_trans - delta_trans)
            s_post = tf.nn.relu(curr_size - processed)
            s_post_sorted.append(s_post)
            imp_post_sorted.append(curr_importance * tf.math.divide_no_nan(s_post, curr_size))
            total_processed_load += processed
            decay = 1.0 - tf.math.exp(-self.lambda_adoi * curr_age)
            total_penalty_reduction += tf.math.divide_no_nan(processed, curr_size) * curr_importance * decay
            if sort_meta is not None and idx < self.param.M - 1:
                sorted_parent_ids = sort_meta[1]
                finish_ratio = tf.where(curr_size > 1e-08, tf.clip_by_value(processed / (curr_size + EPS), 0.0, 1.0), tf.ones_like(curr_size))
                group_success = group_success * finish_ratio
                same_parent = tf.equal(sorted_parent_ids[:, :, idx + 1], sorted_parent_ids[:, :, idx])
                parent_gate = tf.where(same_parent, parent_gate, parent_gate * group_success)
                group_success = tf.where(same_parent, group_success, tf.ones_like(group_success))
        s_post_sorted = tf.stack(s_post_sorted, axis=-1)
        imp_post_sorted = tf.stack(imp_post_sorted, axis=-1)
        if sort_meta is not None:
            inverse_indices = tf.argsort(sort_meta[0], axis=-1, stable=True)
            s_post_window = tf.gather(s_post_sorted, inverse_indices, axis=2, batch_dims=2)
            imp_post_window = tf.gather(imp_post_sorted, inverse_indices, axis=2, batch_dims=2)
        else:
            s_post_window = s_post_sorted
            imp_post_window = imp_post_sorted
        pds_parts = [tf.expand_dims(batch_state[:, :, 0], axis=-1)]
        for idx in range(self.param.M):
            pds_parts.append(tf.expand_dims(s_post_window[:, :, idx] / self.d_ub, axis=-1))
            pds_parts.append(tf.expand_dims(orig_task_ages[:, :, idx], axis=-1))
            pds_parts.append(tf.expand_dims(imp_post_window[:, :, idx], axis=-1))
        pred_start = 1 + self.task_feature_dim * self.param.M
        pred_width = self.param.task_prediction_feature_dim
        if pred_width > 0:
            pds_parts.append(batch_state[:, :, pred_start:pred_start + pred_width])
        log_load = batch_state[:, :, -2]
        log_penalty = batch_state[:, :, -1]
        lin_load = tf.math.expm1(log_load) * self.d_ub
        lin_penalty = tf.math.expm1(log_penalty)
        lin_load_post = tf.nn.relu(lin_load - total_processed_load)
        lin_penalty_post = tf.nn.relu(lin_penalty - total_penalty_reduction)
        pds_parts.append(tf.expand_dims(tf.math.log1p(lin_load_post / self.d_ub), axis=-1))
        pds_parts.append(tf.expand_dims(tf.math.log1p(lin_penalty_post), axis=-1))
        return tf.concat(pds_parts, axis=2)

    def build_pds_exact_batch(self, batch_state, batch_env_action, batch_parent_ids, batch_priorities, batch_importances, batch_trace_idx):
        batch_state = np.asarray(batch_state, dtype=np.float32)
        batch_env_action = np.asarray(batch_env_action, dtype=np.float32)
        batch_parent_ids = np.asarray(batch_parent_ids, dtype=np.int32)
        batch_priorities = np.asarray(batch_priorities, dtype=np.int32)
        batch_importances = np.asarray(batch_importances, dtype=np.float32)
        batch_trace_idx = np.asarray(batch_trace_idx, dtype=np.int32).reshape(-1)
        batch_size = batch_state.shape[0]
        pds_batch = np.zeros_like(batch_state, dtype=np.float32)
        for batch_idx in range(batch_size):
            pds_batch[batch_idx] = self._build_single_pds_exact(batch_state[batch_idx], batch_env_action[batch_idx], batch_parent_ids[batch_idx], batch_priorities[batch_idx], batch_importances[batch_idx], int(batch_trace_idx[batch_idx]))
        return pds_batch

    def _build_single_pds_exact(self, state, env_action, parent_ids, priorities, importances, trace_idx):
        state = np.asarray(state, dtype=np.float32)
        env_action = np.asarray(env_action, dtype=np.float32)
        parent_ids = np.asarray(parent_ids, dtype=np.int32)
        priorities = np.asarray(priorities, dtype=np.int32)
        importances = np.asarray(importances, dtype=np.float32)
        channel = state[:, 0] / 10000000000.0
        f_ratio = np.clip((env_action[:, 0] + 1.0) / 2.0, 0.0, 1.0)
        p_ratio = np.clip((env_action[:, 1] + 1.0) / 2.0, 0.0, 1.0)
        f_vals = f_ratio * np.asarray(self.param.f_max, dtype=np.float32)
        p_vals = p_ratio * np.asarray(self.param.P_max, dtype=np.float32)
        rates = calculate_noma_rate_np(p_vals, channel, float(self.param.sigma2), float(self.param.W_max))
        c_loc = f_vals * float(self.param.d_t) / float(self.param.kappa)
        c_trans = rates * float(self.param.d_t)
        trace_idx = int(trace_idx % self.trace_len)
        load_factor = float(self.workload_trace[trace_idx])
        available_f_edge = float(self.param.F_edge_max) * (1.0 - float(self.param.edge_load_alpha) * load_factor)
        c_edge_total = available_f_edge * float(self.param.d_t) / float(self.param.kappa)
        task_sizes = np.stack([state[:, 1 + self.task_feature_dim * k] for k in range(self.param.M)], axis=-1) * float(self.param.d_ub)
        task_ages = np.stack([state[:, 2 + self.task_feature_dim * k] for k in range(self.param.M)], axis=-1)
        state_importances = np.stack([state[:, 3 + self.task_feature_dim * k] for k in range(self.param.M)], axis=-1)
        x_raw = env_action[:, 2:]
        temp_trans_budgets = c_trans.copy()
        offload_candidates = []
        for user in range(self.param.N):
            for idx in range(self.param.M):
                size = float(task_sizes[user, idx])
                if size <= 0.0:
                    continue
                if x_raw[user, idx] > 0.33 and size <= temp_trans_budgets[user]:
                    offload_candidates.append((int(parent_ids[user, idx]), int(priorities[user, idx]), -size, user, idx, size))
                    temp_trans_budgets[user] -= size
        offload_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        accepted = set()
        current_edge_load = 0.0
        for _, _, _, user, idx, size in offload_candidates:
            if current_edge_load + size <= c_edge_total + 1e-08:
                accepted.add((user, idx))
                current_edge_load += size
        finished = np.zeros((self.param.N, self.param.M), dtype=bool)
        processed_penalty = np.zeros(self.param.N, dtype=np.float32)
        processed_load = np.zeros(self.param.N, dtype=np.float32)
        for user in range(self.param.N):
            tasks = []
            for idx in range(self.param.M):
                size = float(task_sizes[user, idx])
                if size > 0.0:
                    tasks.append((int(parent_ids[user, idx]), int(priorities[user, idx]), -size, idx, size, float(task_ages[user, idx])))
            if not tasks:
                continue
            tasks.sort(key=lambda item: (item[0], item[1], item[2]))
            current_pid = tasks[0][0]
            parent_blocked = False
            budget_loc = float(c_loc[user])
            budget_trans = float(c_trans[user])
            for pid, _, _, idx, size, age in tasks:
                if pid != current_pid:
                    if parent_blocked:
                        break
                    current_pid = pid
                task_finished = False
                decision = float(x_raw[user, idx])
                if decision < -0.33:
                    if size <= budget_loc + 1e-08:
                        budget_loc -= size
                        task_finished = True
                elif decision > 0.33:
                    if size <= budget_trans + 1e-08 and (user, idx) in accepted:
                        budget_trans -= size
                        task_finished = True
                if task_finished:
                    finished[user, idx] = True
                    processed_load[user] += size
                    decay = 1.0 - np.exp(-float(self.param.lambda_adoi) * age)
                    processed_penalty[user] += float(importances[user, idx]) * decay
                else:
                    parent_blocked = True
        pds_state = np.zeros_like(state, dtype=np.float32)
        pds_state[:, 0] = state[:, 0]
        pred_start = 1 + self.task_feature_dim * self.param.M
        pred_width = self.param.task_prediction_feature_dim
        if pred_width > 0:
            pds_state[:, pred_start:pred_start + pred_width] = state[:, pred_start:pred_start + pred_width]
        for user in range(self.param.N):
            remaining_visible = []
            visible_penalty = 0.0
            visible_load = 0.0
            for idx in range(self.param.M):
                size_norm = float(state[user, 1 + self.task_feature_dim * idx])
                if size_norm <= 0.0:
                    continue
                if not finished[user, idx]:
                    age = float(state[user, 2 + self.task_feature_dim * idx])
                    importance = float(state_importances[user, idx])
                    remaining_visible.append((size_norm, age, importance))
                    visible_load += size_norm * float(self.param.d_ub)
                    visible_penalty += importance * (1.0 - np.exp(-float(self.param.lambda_adoi) * age))
            total_load = np.expm1(float(state[user, -2])) * float(self.param.d_ub)
            total_penalty = np.expm1(float(state[user, -1]))
            total_load_post = max(0.0, total_load - processed_load[user])
            total_penalty_post = max(0.0, total_penalty - processed_penalty[user])
            hidden_load = max(0.0, total_load_post - visible_load)
            hidden_penalty = max(0.0, total_penalty_post - visible_penalty)
            pseudo_tasks = self._infer_hidden_slots(hidden_load, hidden_penalty, self.param.M - len(remaining_visible))
            window_tasks = (remaining_visible + pseudo_tasks)[:self.param.M]
            for idx, (size_norm, age, importance) in enumerate(window_tasks):
                pds_state[user, 1 + self.task_feature_dim * idx] = np.float32(size_norm)
                pds_state[user, 2 + self.task_feature_dim * idx] = np.float32(age)
                pds_state[user, 3 + self.task_feature_dim * idx] = np.float32(importance)
            pds_state[user, -2] = np.float32(np.log1p(total_load_post / float(self.param.d_ub)))
            pds_state[user, -1] = np.float32(np.log1p(total_penalty_post))
        return pds_state

    def estimate_cost(self, batch_state, batch_env_action, lam, trace_idx=None, parent_ids=None, priorities=None, importances=None):
        batch_state = tf.cast(batch_state, tf.float32)
        batch_env_action = tf.cast(batch_env_action, tf.float32)
        lam = tf.cast(lam, tf.float32)
        f_ratio = (tf.clip_by_value(batch_env_action[:, :, 0], -1.0, 1.0) + 1.0) / 2.0
        p_ratio = (tf.clip_by_value(batch_env_action[:, :, 1], -1.0, 1.0) + 1.0) / 2.0
        f = f_ratio * self.f_max
        power = p_ratio * self.p_max
        energy = self.gamma_energy * f ** 3 * self.d_t + power * self.d_t
        pds_state = self.build_pds(batch_state, batch_env_action, trace_idx=trace_idx, parent_ids=parent_ids, priorities=priorities, importances=importances)
        adoi_cost = tf.math.expm1(pds_state[:, :, -1])
        total_cost = tf.reduce_sum(adoi_cost + lam[None, :] * tf.nn.relu(energy - tf.convert_to_tensor(self.e_max)), axis=1)
        return total_cost
