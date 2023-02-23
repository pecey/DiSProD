import jax.numpy as jnp
import jax
import numpy as onp
import os
import sys
from jax.test_util import check_grads

DISPROD_PATH = os.getenv("DISPROD_PATH")
sys.path.append(DISPROD_PATH)

from learning.fcn_jax import FCN, derivative_of_silu, second_derivative_of_silu

# Computes the second order derivatives of a vector
def hessian(fn, wrt):
    return jax.jacfwd(jax.jacrev(fn, argnums=wrt), argnums=wrt)
    
# Computes the diagonal of hessian.
# def diag_of_hessian(f, wrt, params):
#     stacked_hessian = hessian(f, wrt)(*params)
#     return jax.numpy.diagonal(stacked_hessian.squeeze([1]), axis1=1, axis2=2)

def diag_of_hessian(f, wrt, params, axis_to_squeeze=None):
    stacked_hessian = hessian(f, wrt)(*params)
    if len(stacked_hessian.shape) == 1:
        return stacked_hessian
    if axis_to_squeeze == None:
        return jax.numpy.diagonal(stacked_hessian, axis1=1, axis2=2)
    return jax.numpy.diagonal(stacked_hessian.squeeze(*axis_to_squeeze), axis1=1, axis2=2)

def forward_pass(model, x):
    z0 = x.reshape(-1, 1)
    common = model.common
    w1, b1 = common[0]
    w2, b2 = common[1]
    w3, b3 = common[2]

    y1 = jnp.matmul(w1, z0) + b1
    z1 = jax.nn.silu(y1)
    y2 = jnp.matmul(w2, z1) + b2
    z2 = jax.nn.silu(y2)
    y3 = jnp.matmul(w3, z2) + b3
    z3 = jax.nn.silu(y3)

    n = x.shape[0]
    cache = {}
    # First order derivatives of the activation fn
    cache["u1"] = derivative_of_silu(y1)
    cache["u2"] = derivative_of_silu(y2)
    cache["u3"] = derivative_of_silu(y3)

    # Second order derivatives of the activation fn
    cache["g1_prime_prime_y1"] = second_derivative_of_silu(y1)
    cache["g2_prime_prime_y2"] = second_derivative_of_silu(y2)
    cache["g3_prime_prime_y3"] = second_derivative_of_silu(y3)

    return z3, cache

def compute_first_order_gradients_fcn(model, n, reusable_params):
    dz0_dxn = jnp.eye(n)
    w1, _ = model.common[0] 
    w2, _ = model.common[1]
    w3, _ = model.common[2]

    def first_partial(dz0_dxi):
        dy1_dxi = jnp.matmul(w1, dz0_dxi.reshape(-1, 1))
        dz1_dxi = jnp.multiply(dy1_dxi, reusable_params["u1"])

        dy2_dxi = jnp.matmul(w2, dz1_dxi)
        dz2_dxi = jnp.multiply(dy2_dxi, reusable_params["u2"])

        dy3_dxi = jnp.matmul(w3, dz2_dxi)
        dz3_dxi = jnp.multiply(dy3_dxi, reusable_params["u3"])

        return dy1_dxi, dy2_dxi, dy3_dxi, dz3_dxi.squeeze()

    dy1_dxn, dy2_dxn, dy3_dxn, dz3_dxn = jax.lax.map(first_partial, dz0_dxn)
    tmp = {}
    tmp["dy1_dxn"] = dy1_dxn
    tmp["dy2_dxn"] = dy2_dxn
    tmp["dy3_dxn"] = dy3_dxn
    return dz3_dxn, tmp

def compute_second_order_gradients_fcn(model, reusable_params, tmp):
    # d2z0_dxn2 = jnp.zeros_like(self.cache["dy1_dxn"])
    w2, _ = model.common[1]
    w3, _ = model.common[2]
    def second_partial(params):
        dy1_dxi, dy2_dxi, dy3_dxi = params
        # u1_dv is zero
        v1_du = jnp.multiply(reusable_params["g1_prime_prime_y1"], jnp.square(dy1_dxi))
        d2z1_dxi2 = v1_du

        u2_dv = jnp.multiply(reusable_params["u2"], jnp.matmul(w2, d2z1_dxi2))
        v2_du = jnp.multiply(reusable_params["g2_prime_prime_y2"], jnp.square(dy2_dxi))
        d2z2_dxi2 = u2_dv + v2_du

        u3_dv = jnp.multiply(reusable_params["u3"], jnp.matmul(w3, d2z2_dxi2))
        v3_du = jnp.multiply(reusable_params["g3_prime_prime_y3"], jnp.square(dy3_dxi))
        d2z3_dxi2 = u3_dv + v3_du
    
        return d2z3_dxi2.squeeze()
    
    dy1_dxn = tmp["dy1_dxn"]
    dy2_dxn = tmp["dy2_dxn"]
    dy3_dxn = tmp["dy3_dxn"]

    return jax.lax.map(second_partial, (dy1_dxn, dy2_dxn, dy3_dxn))

def main():
    # Setup model
    nS = 4
    nA = 2
    n_input = nS + nA
    n_output = nS
    n_hidden = 16
    layer_sizes = [n_input] + [n_hidden]*3 + [n_output]
    model = FCN(layer_sizes = layer_sizes, seed = 42)

    # Setup states and actions
    state = onp.random.random((nS + 1,))
    actions = onp.random.random((nA,))

    # Compute partials using the explicit functions
    noise = state[-1]
    states_wo_noise = state[:-1]
    nS = (states_wo_noise).shape[0]
    nA = actions.shape[0]
    x = jnp.concatenate((states_wo_noise, actions))
    _, reusable_params_vec_1 = model.forward_pass(x)
    _, reusable_params_unvec_1 = forward_pass(model, x)

    # Check if first order gradients are alright
    dcommon_dxn_1, vec_cache = model.compute_first_order_gradients_fcn(reusable_params_vec_1)
    dcommon_dxn_2, unvec_cache = compute_first_order_gradients_fcn(model, x.shape[0], reusable_params_unvec_1)
    dcommon_dxn_autodiff = jnp.squeeze(jax.jacfwd(model.forward_fcn)(x), 1)

    print(f"L-infinity norm for first order partials of FCN with map implementation: {onp.max(onp.abs(dcommon_dxn_1 - dcommon_dxn_2.T))}")
    print(f"L-infinity norm for first order partials of FCN with autodiff: {onp.max(onp.abs(dcommon_dxn_1 - dcommon_dxn_autodiff))}")

    assert onp.allclose(dcommon_dxn_1, dcommon_dxn_2.T, atol=1e-06)

    d2common_dxn2_1 = model.compute_second_order_gradients_fcn(reusable_params_vec_1, vec_cache)
    d2common_dxn2_2 = compute_second_order_gradients_fcn(model, reusable_params_unvec_1, unvec_cache)
    d2common_dxn_autodiff = diag_of_hessian(model.forward_fcn, 0, [x], [1])

    print(f"L-infinity norm for second order partials of FCN with map implementation: {onp.max(onp.abs(d2common_dxn2_1 - d2common_dxn2_2.T))}")
    print(f"L-infinity norm for second order partials of FCN with autodiff: {onp.max(onp.abs(d2common_dxn2_1 - d2common_dxn_autodiff))}")

    assert onp.allclose(d2common_dxn2_1, d2common_dxn2_2.T, atol=1e-06)

    (fop_wrt_state, fop_wrt_action), (sop_wrt_state, sop_wrt_action) = model.get_partials(state, actions)
    # Compute Jacobian of next state
    autodiff_fop_wrt_state = jax.jacfwd(model.next_state_, argnums=0)(states_wo_noise, actions, noise)
    autodiff_fop_wrt_noise = jax.jacfwd(model.next_state_, argnums=2)(states_wo_noise, actions, noise)
    autodiff_fop_wrt_actions = jax.jacfwd(model.next_state_, argnums=1)(states_wo_noise, actions, noise)

    print(f"L-infinity norm for first order partials wrt state with autodiff: {onp.max(onp.abs(autodiff_fop_wrt_state - fop_wrt_state[:, :-1]))}")
    print(f"L-infinity norm for first order partials wrt noise with autodiff: {onp.max(onp.abs(autodiff_fop_wrt_noise - fop_wrt_state[:, -1]))}")
    print(f"L-infinity norm for first order partials wrt action with autodiff: {onp.max(onp.abs(autodiff_fop_wrt_actions - fop_wrt_action))}")

    assert onp.allclose(fop_wrt_state[:, :-1], autodiff_fop_wrt_state, atol=1e-06)
    assert onp.allclose(fop_wrt_action, autodiff_fop_wrt_actions, atol=1e-06)

    # Compute diag of hessian of next state
    autodiff_sop_wrt_state = diag_of_hessian(model.next_state_, 0, [states_wo_noise, actions, noise], None)
    autodiff_sop_wrt_noise = diag_of_hessian(model.next_state_, 2, [states_wo_noise, actions, noise], None)
    autodiff_sop_wrt_actions = diag_of_hessian(model.next_state_, 1, [states_wo_noise, actions, noise], None)

    print(f"L-infinity norm for second order partials wrt state with autodiff: {onp.max(onp.abs(autodiff_sop_wrt_state - sop_wrt_state[:, :-1]))}")
    print(f"L-infinity norm for second order partials wrt noise with autodiff: {onp.max(onp.abs(autodiff_sop_wrt_noise - sop_wrt_state[:, -1]))}")
    print(f"L-infinity norm for second order partials wrt action with autodiff: {onp.max(onp.abs(autodiff_sop_wrt_actions - sop_wrt_action))}")

    assert onp.allclose(sop_wrt_state[:, :-1], autodiff_sop_wrt_state, atol=1e-06)
    assert onp.allclose(sop_wrt_action, autodiff_sop_wrt_actions, atol=1e-06)



if __name__ == "__main__":
    main()

