import jax.numpy as jnp
from jax.example_libraries.optimizers import make_schedule

def adam_with_clipping(step_size, b1=0.9, b2=0.999, eps=1e-8):
  """Construct optimizer triple for Adam with a clip operation

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i, g, state, project_lb, project_ub):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
    mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
    vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
    x =  jnp.clip(x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps), project_lb, project_ub)
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params


def adam_with_projection(step_size, b1=0.9, b2=0.999, eps=1e-8, proj_fn = lambda x: x):
  """Construct optimizer triple for Adam with an added projection step

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
      for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
      for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
      numerical stability (default 1e-8).

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m  # First  moment estimate.
    v = (1 - b2) * jnp.square(g) + b2 * v  # Second moment estimate.
    mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
    vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
    x =  proj_fn(x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps))
    return x, m, v
  def get_params(state):
    x, _, _ = state
    return x
  return init, update, get_params