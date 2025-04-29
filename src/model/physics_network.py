import torch
from tensorflow.core.function.polymorphism.function_type import Parameter
from tensorflow.python.ops.initializers_ns import local_variables

from src.model.NN import Network
import torch.nn as nn
import torch.nn.parameter as parameter

class PhysicsNetwork(nn.Module):
    def __init__(self,
                 cfg,
                 Phi_int,
                 Phi_diff,
                 number_library_terms,
                 function_acceleration,
                 max_values,
    ):
        super(PhysicsNetwork, self).__init__()\
        #untrainable parameters
        self.Phi_int = Phi_int
        self.Phi_diff = Phi_diff
        self.number_library_terms = number_library_terms
        self.function_acceleration = function_acceleration

        #normalized parameters(注册缓冲区avoid the max_values are updated by optimizer)
        self.register_buffer("excitation_max",
                             torch.tensor(max_values["excitation_max"], dtype=torch.float32))
        self.register_buffer("displacement_max",
                             torch.tensor(max_values["displacement_max"], dtype=torch.float32))
        self.register_buffer("velocity_max",
                             torch.tensor(max_values["velocity_max"], dtype=torch.float32))

        #displacement network
        self.displacement_model = Network(cfg)

        #源码是def_coefficient(prefix)
        #coefficient parameters
        self.cx_params = nn.ParameterList([
            Parameter(torch.tensor(1.0), #cx0
            *[Parameter(1.0) for _ in range(number_library_terms - 1)])  #cx1, cx2...
        ])
        #stage label
        self.group_variables_called = False

    def update_function(self, function):
        self.function_acceleration = function

    # cutting coefficient
    def clip_variables(self, threshold=0.10):
        reserve_terms = 3
        for index in range(self.number_library_terms - reserve_terms):
            if torch.abs(self.cx_params[index]) < threshold:
                self.cx_params[index].data.zero_()

    # network or physics variables
    def group_variables(self):
        if not self.group_variables_called:
            self.network_variables = list(self.displacement_model.parameters())
            self.physics_variables = list(self.cx_params)
            self.group_variables_called = True

    # calculate z1_dot by diff_operator * z1
    def predict(self, input):
        # 归一化预测
        normalized_displacement = self.displacement_model(input)
        velocity = torch.matmul(self.Phi_diff, normalized_displacement) * self.displacement_max
        normalized_velocity = velocity / self.velocity_max
        return normalized_displacement, normalized_velocity

    def step_forward(self, input):
        # 获取动态系数
        coeff_dict = {f"cx{idx}": param for idx, param in enumerate(self.cx_params)}

        # predict displacement and velocity
        normalized_displacement, normalized_velocity = self.predict(input)

        f = input.float32() * self.excitation_max
        x = normalized_displacement.float32()
        y = normalized_velocity.float32()

        local_variables = {"f": f, "x": x, "y": y, **coeff_dict}
        acceleration_fit = eval(self.function_acceleration, {}, local_variables)

        # calculate z2 by int_operator * lamda
        integrated_normalized_velcocity = (
            torch.matmul(self.Phi_int, acceleration_fit) / self.velocity_max +
            normalized_velocity[:, 0:1 , :]  #initial velocity
        )
        # velocity error
        normalized_velocity_error = integrated_normalized_velcocity - normalized_velocity
        self.group_variables()

        return normalized_displacement, normalized_velocity_error

    def forward(self, input):
        return  self.step_forward(input)










