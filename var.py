import numpy as np
from scipy.optimize import minimize


"""
Wrapper classes for functional expressions

Copyright (C) 2015-2019

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


"""
Classes:
* Operation
* Expression(Operation)
* Constraint(Expression)
* Array(Expression)
* Variable(Operation)
"""

def sum(arr):
    return Operation.sum(Array(arr))

def max(arr):
    return Operation.max(Array(arr))

def min(arr):
    return Operation.min(Array(arr))

def exp(arr):
    return Operation.exp(Array(arr))

def log(arr):
    return Operation.log(Array(arr))


@np.vectorize
def value_of(variable):
    if isinstance(variable, Operation):
        return variable.value
    elif variable is not None:
        return float(variable)
    else:
        return variable


class Operation:

    #######################
    # basic arithmetic
    #######################
    def __add__(self, right):
        return Expression(self, right, lambda x, y: x + y, '+')

    __radd__ = __add__

    def __sub__(self, right):
        return Expression(self, right, lambda x, y: x - y, '-')

    def __rsub__(self, left):
        return Expression(left, self, lambda x, y: x - y, '-')

    def __mul__(self, right):
        return Expression(right, self, lambda x, y: x * y, '*')

    __rmul__ = __mul__

    def __pow__(self, right):
        return Expression(self, right, lambda x, y: x ** y, '**')

    def __truediv__(self, right):
        return Expression(self, right, lambda x, y: x / y, '/')

    def __rtruediv__(self, left):
        return Expression(left, self, lambda x, y: x / y, '/')

    def __floordiv__(self, right):
        return Expression(self, right, lambda x, y: x // y, '//')

    def __rfloordiv__(self, left):
        return Expression(left, self, lambda x, y: x // y, '//')

    def __matmul__(self, right):
        return Expression(self, right, lambda x, y: x @ y, '@')

    def __rmatmul__(self, left):
        return Expression(left, self, lambda x, y: x @ y, '@')

    #######################
    # unary operations
    #######################
    def __pos__(self):
        return self

    def __neg__(self):
        return Expression(None, self, lambda x, y: -y, lambda x, y: f'-{y}')

    def __abs__(self):
        return Expression(None, self, lambda x, y: abs(y),
                          lambda x, y: f'abs({y})')

    #######################
    # constraints
    #######################
    def __eq__(self, right):
        return Constraint(self, right, lambda x, y: x == y, '==')

    def __le__(self, right):
        return Constraint(self, right, lambda x, y: x <= y, '<=')

    def __lt__(self, right):
        return self.__le__(right)

    def __ge__(self, right):
        return Constraint(self, right, lambda x, y: x >= y, '>=')

    def __gt__(self, right):
        return self.__ge__(right)

    #######################
    # other functions
    #######################
    def sum(self):
        return Expression(None, self, lambda x, y: np.sum(y),
                          lambda x, y: f'sum({y})')

    def min(self):
        return Expression(None, self, lambda x, y: np.min(y),
                          lambda x, y: f'min({y})')

    def max(self):
        return Expression(None, self, lambda x, y: np.max(y),
                          lambda x, y: f'max({y})')

    def exp(self):
        return Expression(None, self, lambda x, y: np.exp(y),
                          lambda x, y: f'exp({y})')

    def log(self):
        return Expression(None, self, lambda x, y: np.log(y),
                          lambda x, y: f'log({y})')

    def abs(self):
        return self.__abs__()

    #######################
    # other
    #######################
    @property
    def value(self):
        raise NotImplementedError


class Expression(Operation):

    def __init__(self, left, right, method, symbol):
        self._left = left
        self._right = right
        self._method = method
        self._symbol = symbol

    def __repr__(self):
        right_name = self._name(self._right)
        left_name = self._name(self._left)
        if isinstance(self._symbol, str):
            return f'({left_name} {self._symbol} {right_name})'
        else:
            return self._symbol(left_name, right_name)

    @staticmethod
    def _name(var):
        if var is not None:
            return var if not hasattr(var, 'name') else var.name
        else:
            return ''

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def value(self):
        left_val = value_of(self._left)
        right_val = value_of(self._right)
        return self._method(left_val, right_val)

    def variables(self):

        # helper function
        def update(value, variables, expressions):
            if isinstance(value, Array):
                variables = variables.union([v.id for v in value.variables()])
            elif isinstance(value, Expression):
                expressions.append(value)
            elif isinstance(value, Variable):
                variables.add(value.id)
            return variables, expressions

        # initialise
        list_expr = [self]
        list_var = set()

        while len(l_expr) > 0:

            expression = l_expr.pop()
            list_var, list_expr = update(expression.left, list_var, list_expr)
            list_var, list_expr = update(expression.right, list_var, list_expr)

        variables = [Variable._vars[i] for i in list_var]
        return variables


class Constraint(Expression):

    @property
    def is_equality(self):
        return self._symbol == '=='

    @property
    def is_greater_than(self):
        return self._symbol == '>='

    @property
    def is_less_than(self):
        return self._symbol == '<='


class Array(Expression):
    def __init__(self, value):
        self._shape = np.shape(value)
        self._value = np.array(value)

    def __repr__(self):
        value = np.copy(self._value)
        value_ravelled = value.ravel()
        original_value_ravelled = self._value.ravel()
        for i in range(len(original_value_ravelled)):
            val = original_value_ravelled[i]
            if isinstance(val, Variable):
                value_ravelled[i] = val.name
            else:
                value_ravelled[i] = val

        s = value.__repr__()
        s = s.replace('array', '').replace(', dtype=object', '')
        s = '\n'.join(x.strip() for x in s.split('\n'))
        return s[1:-1]

    def __getitem__(self, key):
        return self._value[key]

    @property
    def shape(self):
        return self._shape

    @property
    def value(self):
        return value_of(self._value)

    def variables(self):
        variables = []
        for var in self._value.flatten():
            if isinstance(var, Variable):
                variables.append(var)
            elif isinstance(var, Expression):
                variables.extend(var.variables())
        return variables


class Variable(Operation):

    _n = 0
    _initial_guesses = []
    _x = []
    _vars = []

    def __init__(self, name=None, value=None):
        self.value = value
        self._id = Variable._n
        self._name = f'var{self._id}' if name is None else name
        # TODO: add variables that are not just scalars

        Variable._n += 1
        Variable._vars.append(self)

    def __repr__(self):
        if self.value:
            return f'<{self._name}: {self.value:.4f}>'
        else:
            return f'<{self._name}>'

    def __str__(self):
        return f'<{self._name}: {self.value:.4f}>'

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class Problem:

    def __init__(self, obj, constraints=None, tol=1e-6):
        self._obj = obj
        self._constraints = constraints
        self._variables = obj.variables()
        self._obj_fun = self._build_objective_function(obj)
        if isinstance(constraints, Constraint):
            constraints = [constraints]
        elif constraints is None:
            constraints = []
        self._constraint_funs = self._build_constraints(constraints)

    def __repr__(self):
        return (f'minimise {self._obj}\n' +
                's.t.\n' +
                '\n'.join([f'{c}' for c in self._constraints]))

    def _assign_to_variables(self, x):
        for i, var in enumerate(self._variables):
            var.value = x[i]

    def _initial_guess(self):
        return [1. if x.value is None else x.value
                for x in self._variables]

    def minimize(self):
        opt = minimize(self._obj_fun, self._initial_guess(),
                       constraints=self._constraint_funs)
        self._assign_to_variables(opt.x)
        return opt

    def _build_objective_function(self, expression):
        def fun(x):
            self._assign_to_variables(x)
            return expression.value
        return fun

    def _build_constraints(self, constraints):
        new = []
        for constraint in constraints:
            if constraint.is_equality:
                def fun(x, constraint=constraint):
                    self._assign_to_variables(x)
                    return (constraint._left - constraint._right).value
                new.append({'type': 'eq', 'fun': fun})
            elif constraint.is_greater_than:
                def fun(x, constraint=constraint):
                    self._assign_to_variables(x)
                    return (constraint._left - constraint._right).value
                new.append({'type': 'ineq', 'fun': fun})
            elif constraint.is_less_than:
                def fun(x, constraint=constraint):
                    self._assign_to_variables(x)
                    return (constraint._right - constraint._left).value
                new.append({'type': 'ineq', 'fun': fun})
        return new


if __name__ == '__main__':
    # fake data
    a = 2
    m = 3
    x = np.linspace(0, 10)
    y = a * x + m + np.random.randn(len(x))

    a_ = Variable()
    m_ = Variable()
    y_ = a_ * x + m_
    error = y_ - y
    prob = Problem((error**2).sum(), None)
    prob.minimize()
    print(f'a = {a}, a_ = {a_}')
    print(f'm = {m}, m_ = {m_}')
