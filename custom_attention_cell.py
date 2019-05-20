from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.ops.math_ops import sigmoid,tanh
from tensorflow.python.ops import array_ops,clip_ops,embedding_ops,init_ops,math_ops,nn_ops,partitioned_variables,variable_scope as vs
import collections
import math


class customGRUCellWithAttention(RNNCell):

    def __init__(self, num_units, input_size=None, activation=tanh):
        self.num_units = num_units
        self.activation = activation

    @property
    def state_size(self):
        return self.num_units


    @property
    def output_size(self):
        return self.num_units

    def __call__(self, inputs, state, scope=None):
        """Attention GRU with nunits cells."""
        with vs.variable_scope("attention_gru_cell"):
            with vs.variable_scope("GRUgates"):

                model_inputs, g = array_ops.split(inputs,num_or_size_splits=[self.num_units,1],axis=1)
                rect_linear = func_linear([model_inputs, state], self.num_units, True)
                rect_linear = sigmoid(rect_linear)
            with vs.variable_scope("candidate"):
                rect_linear = rect_linear*func_linear(state, self.num_units, False)
            with vs.variable_scope("input"):
                x = func_linear(model_inputs, self.num_units, True)
            h_hat = self.activation(rect_linear + x)

            h = (1 - g) * state + g * h_hat
        return h, h

def func_linear(args, output_size, bias, bias_start=0.0):

    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            "weights", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable(
                        "biases", [output_size],
                      dtype=dtype,
                    initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
        return nn_ops.bias_add(res, biases)
