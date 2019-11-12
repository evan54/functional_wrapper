import numpy as np
from scipy.optimize import minimize


def sum(self):
    return Operation.sum(self)


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
    # other
    #######################
    def sum(self):
        return Expression(None, self, lambda x, y: np.sum(y),
                          lambda x, y: f'sum({y})')

    @property
    def value(self):
        raise NotImplementedError


class Expression(Operation):

    def __init__(self, left, right, method, symbol):
        self._left = left
        self._right = right
        self._method = method
        self._symbol = symbol

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @staticmethod
    @np.vectorize
    def _get_value_of(variable):
        if isinstance(variable, Operation):
            return variable.value
        elif variable is not None:
            return float(variable)
        else:
            return variable

    @property
    def value(self):
        left_val = self._get_value_of(self._left)
        right_val = self._get_value_of(self._right)
        return self._method(left_val, right_val)

    @staticmethod
    def _name(var):
        if var is not None:
            return var if not hasattr(var, 'name') else var.name
        else:
            return ''

    def _str_expression(self):
        right_name = self._name(self._right)
        left_name = self._name(self._left)
        if isinstance(self._symbol, str):
            return f'({left_name} {self._symbol} {right_name})'
        else:
            return self._symbol(left_name, right_name)

    def __repr__(self):
        return self._str_expression()

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
        l_expr = [self]
        l_var = set()

        while len(l_expr) > 0:

            expression = l_expr.pop()
            l_var, l_expr = update(expression.left, l_var, l_expr)
            l_var, l_expr = update(expression.right, l_var, l_expr)

        variables = [Variable._vars[i] for i in l_var]
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

    @property
    def shape(self):
        return self._shape

    @property
    def value(self):
        return self._get_value_of(self._value)

    def __repr__(self):
        return self._value.__repr__()

    def variables(self):
        variables = []
        for var in self._value.flatten():
            if isinstance(var, Variable):
                variables.append(var)
        return variables

    def __getitem__(self, key):
        return self._value[key]


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

    def __repr__(self):
        return f'<{self._name}: {self.value}>'

    def __str__(self):
        return f'<{self._name}: {self.value}>'


class Problem:

    def __init__(self, obj, constraints=None, tol=1e-6):
        self._variables = obj.variables()
        self._obj = self._build_objective_function(obj)
        if isinstance(constraints, Constraint):
            constraints = [constraints]
        elif constraints is None:
            constraints = []
        self._constraints = self._build_constraints(constraints)

    def _assign_to_variables(self, x):
        for i, var in enumerate(self._variables):
            var.value = x[i]

    def _initial_guess(self):
        return [1. if x.value is None else x.value
                for x in self._variables]

    def minimize(self):
        opt = minimize(self._obj, self._initial_guess(),
                       constraints=self._constraints)
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
