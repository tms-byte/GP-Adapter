import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import LinearKernel, RBFKernel
from gpytorch.means import ZeroMean
from tqdm import tqdm


class LocalGPImage(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, tau2_mle=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean(batch_shape=torch.Size([num_classes]))
        self.covar_module1 = RBFKernel(batch_shape=torch.Size([num_classes]))
        self.covar_module1.variance = tau2_mle

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module1(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LocalGPText(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, tau2_mle=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean(batch_shape=torch.Size([num_classes]))
        self.covar_module1 = LinearKernel(
            batch_shape=torch.Size([num_classes]),
        )
        self.covar_module1.variance = tau2_mle  # ここに MLE 値をセット

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module1(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RBFKernelOptimizer:
    def __init__(self, X, y, length_scales=None, loss_thr=-10):
        self.X = X.to(dtype=torch.float32).cpu()
        self.y = y.to(dtype=torch.float32).cpu()
        self.num_classes = X.size(0)
        self.num_shots = X.size(1)
        self.length_scales = length_scales if length_scales is not None else torch.arange(0.1, 2.05, 0.05)
        self.loss_thr = loss_thr
        self.results = {
            'best_length_scale': torch.zeros(self.num_classes),
            'best_tau2': torch.zeros(self.num_classes),
            'best_log_likelihood': torch.zeros(self.num_classes),
        }

    def compute_classwise_optima(self):
        for c in tqdm(range(self.num_classes)):
            xc = self.X[c]
            yc = self.y[c]

            best_ll = -float('inf')
            best_ls = None
            best_tau2 = None

            for ls in self.length_scales:
                kernel = gpytorch.kernels.RBFKernel()
                kernel.lengthscale = ls

                k0 = kernel(xc, xc).evaluate()
                k0 += 1e-6 * torch.eye(k0.size(-1))

                try:
                    k0_inv = torch.linalg.inv(k0)
                except RuntimeError:
                    continue

                numerator = torch.matmul(yc.view(1, -1), torch.matmul(k0_inv, yc.view(-1, 1))).squeeze()
                tau2_mle = numerator / self.num_shots

                term1 = -0.5 * numerator
                term2 = -0.5 * torch.logdet(k0)
                term3 = -0.5 * self.num_shots * torch.log(torch.tensor(2 * torch.pi))
                log_likelihood = (term1 + term2 + term3).item()

                if log_likelihood >= self.loss_thr:
                    continue

                if log_likelihood > best_ll:
                    best_ll = log_likelihood
                    best_ls = ls
                    best_tau2 = tau2_mle

            self.results['best_length_scale'][c] = best_ls if best_ls is not None else -1
            self.results['best_tau2'][c] = best_tau2 if best_tau2 is not None else -1
            self.results['best_log_likelihood'][c] = best_ll

        return self.results
