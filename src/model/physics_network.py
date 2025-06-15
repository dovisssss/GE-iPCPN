import torch
import torch.nn as nn
import re
import sympy as sp
import numpy as np
from scipy.signal import savgol_filter

from src.model.NN import Network
#import torch.nn.parameter as Parameter
from src.model.function_library import FunctionLibrary as Library
from src.model.grad_function_lib import gradFunctionLibrary as GradFunctionLibrary
from src.data.numercial_operator import sg_filter

class PhysicsNetwork(nn.Module):
    def __init__(self,
                 cfg,
                 Phi_int,
                 Phi_diff,
                 number_library_terms,
                 number_grad_library_terms,
                 function_acceleration,
                 grad_expression,
                 max_values,
    ):
        super(PhysicsNetwork, self).__init__()
        #untrainable parameters
        self.Phi_int = Phi_int
        self.Phi_diff = Phi_diff
        self.number_library_terms = number_library_terms
        self.number_grad_library_terms = number_grad_library_terms
        self.function_acceleration = function_acceleration
        self.grad_expression = grad_expression
        self.library = Library()
        self.grad_library = GradFunctionLibrary()

        #normalized parameters(注册缓冲区avoid the max_values are updated by optimizer)
        self.register_buffer("excitation_max",
                             torch.tensor(max_values["excitation_max"], dtype=torch.float32))
        self.register_buffer("displacement_max",
                             torch.tensor(max_values["displacement_max"], dtype=torch.float32))
        self.register_buffer("velocity_max",
                             torch.tensor(max_values["velocity_max"], dtype=torch.float32))

        self.displacement_max = torch.tensor(max_values["displacement_max"], dtype=torch.float32)
        self.velocity_max = torch.tensor(max_values["velocity_max"], dtype=torch.float32)
        self.excitation_max = torch.tensor(max_values["excitation_max"], dtype=torch.float32)


        #displacement network
        self.displacement_model = Network(cfg)

        #coefficient parameters
        param = [nn.Parameter(torch.tensor(1.0))]
        param += [nn.Parameter(torch.tensor(1.0)) for _ in range(number_library_terms-1)]
        #梯度方程单独的sindy参数
        param_grad = [nn.Parameter(torch.tensor(1.0))]
        param_grad += [nn.Parameter(torch.tensor(1.0)) for _ in range(number_grad_library_terms-1)]
        self.cx_params = nn.ParameterList(param)
        self.grad_params = nn.ParameterList(param_grad)
        #stage label
        self.group_variables_called = False

        self.window_size = 21
        self.poly_order = 2
        self.dt = 1/2000

    #def network_params(self):
    #    return list(self.displacement_model.parameters())

    #def physics_params(self):
    #    return list(self.cx_params)

    def update_function(self, function, grad_expression):
        self.function_acceleration = function
        self.grad_expression = grad_expression

    # cutting coefficient
    def clip_variables(self, threshold=0.10):
        reserve_terms = 3
        for index in range(self.number_library_terms - reserve_terms):
            if torch.abs(self.cx_params[index]) < threshold:
                self.cx_params[index].data.zero_()
        for index in range(self.number_grad_library_terms - reserve_terms):
            if torch.abs(self.grad_params[index]) < threshold:
                self.grad_params[index].data.zero_()

    # network or physics variables
    def group_variables(self):
        if not self.group_variables_called:
            self.network_variables = list(self.displacement_model.parameters())
            self.physics_variables = list(self.cx_params)
            self.physics_grad_variables = list(self.grad_params)
            self.group_variables_called = True
        return self.network_variables, self.physics_variables, self.physics_grad_variables

    # calculate z1_dot by diff_operator * z1
    def predict(self, input):
        # Normalized prediction
        normalized_displacement = self.displacement_model(input)
        velocity = torch.matmul(self.Phi_diff, normalized_displacement) * self.displacement_max
        normalized_velocity = velocity / self.velocity_max
        return normalized_displacement, normalized_velocity

    def step_forward(self, input):
        # Get coefficient
        coeff_dict = {f"cx{idx}": param for idx, param in enumerate(self.cx_params)}
        grad_coeff_dict = {f"ge{idx}": param for idx, param in enumerate(self.grad_params)}
        # predict displacement and velocity

        normalized_displacement, normalized_velocity = self.predict(input)

        f_value = input.float() * self.excitation_max
        x_value = normalized_displacement.float()
        y_value = normalized_velocity.float()

        local_variables = {"f": f_value, "x": x_value, "y": y_value, **coeff_dict, "torch":torch}
        acceleration_fit = eval(self.function_acceleration, {'torch': torch}, local_variables)

        # calculate z2 by int_operator * lamda
        integrated_normalized_velocity = (
            torch.matmul(self.Phi_int, acceleration_fit) / self.velocity_max +
            normalized_velocity[:, 0:1 , :]  #initial velocity
        )
        # velocity error
        normalized_velocity_error = integrated_normalized_velocity - normalized_velocity

        # gradient equation error
        acc_grad = sg_filter(acceleration_fit, self.window_size, self.poly_order, deriv=1,  axis=1)/self.dt#/4286.548#*18.755093

        f_grad = sg_filter(f_value, self.window_size, self.poly_order, deriv=1, axis=1)/self.dt
        v = sg_filter(x_value, self.window_size, self.poly_order, deriv=1, axis=1)/self.dt
        a = sg_filter(y_value, self.window_size, self.poly_order, deriv=1, axis=1)/self.dt

        gard_variables = {"f": f_grad, "x": x_value, "y": y_value, "v":v, "a":a, **grad_coeff_dict, "torch": torch}
        acc_grad_fit = eval(self.grad_expression, {'torch': torch}, gard_variables)

        acc_error = (acc_grad_fit - acc_grad)*10e-3

        self.group_variables()

        return normalized_displacement, normalized_velocity_error, acc_error

    def forward(self, input):
        return  self.step_forward(input)










