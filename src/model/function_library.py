class FunctionLibrary:
    def __init__(self):
        self.library_terms = [
            #Special term
            "f",
            # Degree 1
            "x",
            "y",
            # Degree 2
            "torch.pow(x , 2)",
            "x * y",
            "torch.pow(y , 2)",
            # Degree 3
            "torch.pow(x , 3)",
            "torch.pow(x , 2) * y",
            "x * torch.pow(y , 2)",
            "torch.pow(y , 3)",
            # Degree 4
            "torch.pow(x , 4)",
            "torch.pow(x , 3) * y",
            "torch.pow(x , 2) * torch.pow(y , 2)",
            "x * torch.pow(y , 3)",
            "torch.pow(y , 4)",
            # Degree 5
            "torch.pow(x , 5)",
            "torch.pow(x , 4) * y",
            "torch.pow(x , 3) * torch.pow(y , 2)",
            "torch.pow(x , 2) * torch.pow(y , 3)",
            "x * torch.pow(y , 4)",
            "torch.pow(y , 5)",
        ]
        self.terms_number = len(self.library_terms)

    def build_functions(self, coefficients):
        """Build the functions for acceleration.
        Args:
            lambda (list): List of coefficients for x.
        Returns:
            function (str): Function for x.
        """
        function_x = ""
        for i in range(self.terms_number):
            term = self.library_terms[i]
            if coefficients[i] != 0:
                function_x += f"+cx{i}*{term}"    #cx[i]意义？
        function_x = function_x[1:]
        return function_x

    def get_functions(self, coefficients):
        function_x = ""
        for i in range(self.terms_number):
            term = self.library_terms[i]
            if coefficients[i] != 0:
                function_x += f"+{coefficients[i].detach().cpu().numpy():.4f}*{term}"
        function_x = function_x[1:].replace("+-" , "-")
        return function_x
