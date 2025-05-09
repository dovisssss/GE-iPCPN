import torch

from src.model.NN import Network
import torch.nn as nn
#import torch.nn.parameter as Parameter

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

        self.displacement_max = torch.tensor(max_values["displacement_max"], dtype=torch.float32)
        self.velocity_max = torch.tensor(max_values["velocity_max"], dtype=torch.float32)
        self.excitation_max = torch.tensor(max_values["excitation_max"], dtype=torch.float32)


        #displacement network
        self.displacement_model = Network(cfg)

        #coefficient parameters
        param = [nn.Parameter(torch.tensor(1.0))]
        param += [nn.Parameter(torch.tensor(1.0)) for _ in range(number_library_terms-1)]
        self.cx_params = nn.ParameterList(param)
        #stage label
        self.group_variables_called = False

    #def network_params(self):
    #    return list(self.displacement_model.parameters())

    #def physics_params(self):
    #    return list(self.cx_params)

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
        return self.network_variables, self.physics_variables

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

        # predict displacement and velocity
        normalized_displacement, normalized_velocity = self.predict(input)

        f = input.float() * self.excitation_max
        x = normalized_displacement.float()
        y = normalized_velocity.float()

        local_variables = {"f": f, "x": x, "y": y, **coeff_dict, "torch":torch}
        acceleration_fit = eval(self.function_acceleration, {'torch': torch}, local_variables)

        # calculate z2 by int_operator * lamda
        integrated_normalized_velocity = (
            torch.matmul(self.Phi_int, acceleration_fit) / self.velocity_max +
            normalized_velocity[:, 0:1 , :]  #initial velocity
        )
        # velocity error
        normalized_velocity_error = integrated_normalized_velocity - normalized_velocity
        self.group_variables()

        return normalized_displacement, normalized_velocity_error

    def forward(self, input):
        return  self.step_forward(input)










