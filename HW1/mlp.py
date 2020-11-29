import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # DONE: Implement the forward function
        self.cache["x"] = x
        self.cache["z1"] = torch.matmul(x, self.parameters["W1"].T) + self.parameters["b1"]

        if self.f_function == "relu":
          self.cache["z2"] = self.relu(self.cache["z1"])
        elif self.f_function == "sigmoid":
          self.cache["z2"] = self.sigmoid(self.cache["z1"])
        elif self.f_function == "identity":
          self.cache["z2"] = self.cache["z1"]
        
        self.cache["z3"] = torch.matmul(self.cache["z2"], self.parameters["W2"].T) + self.parameters["b2"]

        if self.g_function == "relu":
          self.cache["y_hat"] = self.relu(self.cache["z3"])
        elif self.g_function == "sigmoid":
          self.cache["y_hat"] = self.sigmoid(self.cache["z3"])
        elif self.g_function == "identity":
          self.cache["y_hat"] = self.cache["z3"]

        return self.cache["y_hat"]
    
    def relu(self, z):
      return z.clamp(min=0)

    def sigmoid(self, z):
      return 1 / (1+torch.exp(-z))
    
    def sigmoid_backward(self, z):
      return z * (1-z)

    def relu_backward(self, z):
      return torch.where(z <= 0, torch.zeros(z.shape), torch.ones(z.shape))

    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # DONE: Implement the backward function
        if self.g_function == "sigmoid":
          self.cache["dyhat_dz3"] = self.sigmoid_backward(self.cache["y_hat"])
        elif self.g_function == "relu":
          self.cache["dyhat_dz3"] = self.relu_backward(self.cache["z3"])
        elif self.g_function == "identity":
          self.cache["dyhat_dz3"] = torch.ones(self.cache["z3"].shape)

        self.cache["dJ_dz3"] = dJdy_hat * self.cache["dyhat_dz3"]/dJdy_hat.shape[0]
        self.grads["dJdW2"] = torch.matmul(self.cache["dJ_dz3"].T, self.cache["z2"])
        self.grads["dJdb2"] = torch.matmul(self.cache["dJ_dz3"].T, torch.ones(dJdy_hat.shape[0]))

        if self.f_function == "sigmoid":
            self.cache["dz2_dz1"] = self.sigmoid_backward(self.cache["z2"])
        elif self.f_function == "relu":
          self.cache["dz2_dz1"] = self.relu_backward(self.cache["z1"])
        elif self.f_function == "identity":
          self.cache["dz2_dz1"] = torch.ones(self.cache["z1"].shape)

        self.cache["dJ_dz2"] = torch.matmul(self.cache["dJ_dz3"], self.parameters["W2"])
        self.cache["dJ_dz1"] = self.cache["dJ_dz2"] * self.cache["dz2_dz1"]

        self.grads["dJdW1"] = torch.matmul(self.cache["dJ_dz1"].T, self.cache["x"])
        self.grads["dJdb1"] = torch.matmul(self.cache["dJ_dz1"].T, torch.ones(self.cache["dJ_dz1"].shape[0]))
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # DONE: Implement the mse loss
    loss = 0.5 * torch.mean((y-y_hat)**2)
    dJdy_hat = - y + y_hat

    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # DONE: Implement the bce loss
    loss = (-torch.mul(y, torch.log(y_hat)) - torch.mul(1-y, torch.log(1-y_hat))).mean()
    dJdy_hat = -y/y_hat + (1-y)/(1-y_hat)

    return loss, dJdy_hat











