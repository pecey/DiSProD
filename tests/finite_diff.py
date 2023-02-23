from learning.fcn_torch import FCN
import torch as T
import torch.nn.functional as F
from utils import second_derivative_of_swish

def compute_finite_derivatives(model, states, actions, n):
    with T.no_grad():
        y = model.next_state(states, actions)
        epsilon = T.rand(1)[0]/100
        if n < 4:
            state_prime_1 = T.clone(states)
            state_prime_2 = T.clone(states)
            state_prime_1[n][0] += epsilon
            state_prime_2[n][0] -= epsilon
            action_prime_1 = actions
            action_prime_2 = actions
        else:
            action_prime_1 = T.clone(actions)
            action_prime_2 = T.clone(actions)
            action_prime_1[n-4][0] += epsilon
            action_prime_2[n-4][0] -= epsilon
            state_prime_1 = states
            state_prime_2 = states

        y_epsilon_1 = model.next_state(state_prime_1, action_prime_1)
        y_epsilon_2 = model.next_state(state_prime_2, action_prime_2)

        # f(x+e)-f(x)/e
        first_order_grad = (y_epsilon_1 - y)/epsilon

        # f(x+e)-2f(x)+f(x-e)/e^2
        second_order_grad = (y_epsilon_1 - 2*y + y_epsilon_2)/T.square(epsilon)

        return first_order_grad, second_order_grad

def compute_finite_derivatives_common(model, states, actions, n):
    T.set_printoptions(precision=8, sci_mode=False)
    with T.no_grad():
        y = model.next_state_common(states, actions)
        e = T.rand(1)[0]/100
        if n < 4:
            state_plus_e = T.clone(states)
            state_minus_e = T.clone(states)
            state_plus_e[n][0] += e
            state_minus_e[n][0] -= e
            action_plus_e = actions
            action_minus_e = actions
        else:
            action_plus_e = T.clone(actions)
            action_minus_e = T.clone(actions)
            action_plus_e[n-4][0] += e
            action_minus_e[n-4][0] -= e
            state_plus_e = states
            state_minus_e = states

        y_plus_e = model.next_state_common(state_plus_e, action_plus_e)
        y_minus_e = model.next_state_common(state_minus_e, action_minus_e)

        # f(x+e)-f(x)/e
        first_order_grad = (y_plus_e - y)/e

        # f(x+e)-2f(x)+f(x-e)/e^2
        second_order_grad = (y_plus_e - 2*y + y_minus_e)/T.square(e)

        return first_order_grad, second_order_grad


def main():
    T.manual_seed(42)
    model = FCN(6, 32, 4, 'cpu')
    states = T.rand(5).reshape(-1, 1)
    actions = T.rand(2).reshape(-1, 1)

    state_without_noise = states[:-1, :]
    x = T.cat((state_without_noise, actions)).float()
    model.forward_pass(x.T, 6)

    # first_order_symbolic_derivatives = model.next_state_first_order_partials(states, actions, 7).T
    # second_order_symbolic_derivatives = model.next_state_second_order_partials(states, actions, 7).T
    # numerical_derivatives = [compute_finite_derivatives(model, states, actions, n) for n in range(6)]

    first_order_symbolic_derivatives = model.next_state_first_order_partials(states, actions, 7).T
    second_order_symbolic_derivatives = model.next_state_second_order_partials(states, actions, 7).T
    numerical_derivatives = [compute_finite_derivatives(model, states, actions, n) for n in range(6)]
    print(first_order_symbolic_derivatives.shape)
    print(second_order_symbolic_derivatives.shape)
    print(numerical_derivatives[0][0].shape)
    print(numerical_derivatives[0][1].shape)


    print("### First order derivatives ###")
    for i in range(4):
        print(f"Derivative wrt state {i}")
        print(f"Symbolic : {first_order_symbolic_derivatives[i]}")
        print(f"Numeric: {numerical_derivatives[i][0].flatten()}")
    for i in range(2):
        print(f"Derivative wrt action {i}")
        print(f"Symbolic : {first_order_symbolic_derivatives[5+i]}")
        print(f"Numeric: {numerical_derivatives[4+i][0].flatten()}")

    print("### Second order derivatives ###")
    for i in range(4):
        print(f"Derivative wrt state {i}")
        print(f"Symbolic : {second_order_symbolic_derivatives[i]}")
        print(f"Numeric: {numerical_derivatives[i][1].flatten()}")
    for i in range(2):
        print(f"Derivative wrt action {i}")
        print(f"Symbolic : {second_order_symbolic_derivatives[5+i]}")
        print(f"Numeric: {numerical_derivatives[4+i][1].flatten()}")



if __name__ == "__main__":
    main()
