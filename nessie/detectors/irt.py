from typing import Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import AdagradRMSProp
from tqdm import trange

from nessie.detectors.error_detector import Detector, DetectorKind


class ItemResponseTheoryFlagger(Detector):
    """Evaluation Examples Are Not Equally Informative: How Should That Change NLP Leaderboards?
    Pedro Rodriguez and Joe Barrow and Alexander Hoyle and John P. Lalor and Robin Jia and Jordan Boyd-Graber
    ACL 2021

    https://research.fb.com/wp-content/uploads/2021/07/Evaluation-Examples-Are-Not-Equally-Informative-How-Should-That-Change-NLP-Leaderboards.pdf
    """

    def __init__(self, device: str = "cpu", num_iters: int = 10_000):
        self._device = device
        self._num_iters = num_iters

    def score(self, labels: np.ndarray, ensemble_predictions: np.ndarray, **kwargs) -> np.ndarray:
        """Flags instances with negative discrimination as computed by an IRT model. This is typically applied to the
        predictions of several different models, similarly to ensembling.

        Args:
            labels: a (num_samples, ) numpy array containing the noisy labels to be corrected
            ensemble_predictions: a (num_models, num_samples) numpy array containing predictions for each model

        Returns:
            a (num_samples, ) numpy array containing flagging instances having negative discrimination
        """

        # https://github.com/facebookresearch/irt-leaderboard/blob/ac864bfb7145cb65353dab232f331552cc82a72f/leaderboard/irt/model_svi.py

        data = (ensemble_predictions == labels).astype(int)
        assert len(labels) == ensemble_predictions.shape[1]

        assert data.shape == ensemble_predictions.shape

        # IRT likes num_samples in the first dimension, therefore, we transpose it
        data = data.T

        subjects, items, responses = self.convert_data(data)
        num_items, num_subjects = data.shape

        # if running_in_slurm():
        #     girth_model = GirthMCMC(
        #         model="2PL", options={"variational_inference": True, "variational_samples": 25000, "n_samples": 25000}
        #     )
        #     results = girth_model(data)
        # else:
        #     results = twopl_mml(data)

        subjects, items, responses = self.convert_data(data)
        num_items, num_subjects = data.shape

        model = TwoParamLog(num_items=num_items, num_subjects=num_subjects, device=self._device)

        self.optimize(model.get_model(), model.get_guide(), items, subjects, responses)
        result = model.export()
        disc_svi = np.array(result["disc"])

        return (disc_svi < 0.0).astype(bool)

    def error_detector_kind(self) -> DetectorKind:
        return DetectorKind.FLAGGER

    def convert_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_items, num_subjects = data.shape

        # questions[i] holds the id of the participant for observation i
        questions = np.repeat(np.arange(num_items), num_subjects).astype(np.int64)

        # subjects[i] holds the id of the participant for observation i
        subjects = np.tile(np.arange(num_subjects), num_items).astype(np.int64)

        # correctnesses[i] holds 1 if observation i was correct else 0
        correctnesses = data.flatten().astype(float)

        return subjects, questions, correctnesses

    def optimize(self, model, guide, items: np.ndarray, subjects: np.ndarray, correctnesses: np.ndarray):
        num_epochs = self._num_iters

        optimizer = pyro.optim.Adam({"lr": 0.001})

        initial_lr = 0.001
        gamma = 0.1  # final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / num_epochs)
        optim = pyro.optim.ClippedAdam({"lr": initial_lr, "lrd": lrd})

        svi = SVI(
            model,
            guide,
            optimizer,
            loss=Trace_ELBO(),
        )

        pyro.clear_param_store()

        pbar = trange(num_epochs)

        subjects_tensor = torch.from_numpy(subjects).long().to(self._device)
        items_tensor = torch.from_numpy(items).long().to(self._device)
        correctnesses_tensor = torch.from_numpy(correctnesses).float().to(self._device)

        for j in pbar:
            loss = svi.step(subjects_tensor, items_tensor, correctnesses_tensor)
            if j % 100 == 0:
                pbar.set_postfix_str("[epoch %04d] loss: %.4f" % (j + 1, loss))

    def uses_probabilities(self) -> bool:
        return True


class TwoParamLog:
    """2PL IRT model taken from https://github.com/nd-ball/py-irt"""

    def __init__(self, *, num_items: int, num_subjects: int, device: str = "cpu"):
        self.num_items = num_items
        self.num_subjects = num_subjects
        self.device = device

    def model_hierarchical(self, subjects, items, obs):
        """Initialize a 2PL model with hierarchical priors"""
        mu_b = pyro.sample(
            "mu_b",
            dist.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device)),
        )
        u_b = pyro.sample(
            "u_b",
            dist.Gamma(torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)),
        )
        mu_theta = pyro.sample(
            "mu_theta",
            dist.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device)),
        )
        u_theta = pyro.sample(
            "u_theta",
            dist.Gamma(torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)),
        )
        mu_a = pyro.sample(
            "mu_a",
            dist.Normal(torch.tensor(0.0, device=self.device), torch.tensor(1.0e6, device=self.device)),
        )
        u_a = pyro.sample(
            "u_a",
            dist.Gamma(torch.tensor(1.0, device=self.device), torch.tensor(1.0, device=self.device)),
        )
        with pyro.plate("thetas", self.num_subjects, device=self.device):
            ability = pyro.sample("theta", dist.Normal(mu_theta, 1.0 / u_theta))
        with pyro.plate("bs", self.num_items, device=self.device):
            diff = pyro.sample("b", dist.Normal(mu_b, 1.0 / u_b))
            slope = pyro.sample("a", dist.Normal(mu_a, 1.0 / u_a))
        with pyro.plate("observe_data", obs.size(0)):
            pyro.sample(
                "obs",
                dist.Bernoulli(logits=slope[items] * (ability[subjects] - diff[items])),
                obs=obs,
            )

    def guide_hierarchical(self, subjects, items, obs):
        """Initialize a 2PL guide with hierarchical priors"""
        loc_mu_b_param = pyro.param("loc_mu_b", torch.tensor(0.0, device=self.device))
        scale_mu_b_param = pyro.param(
            "scale_mu_b", torch.tensor(1.0e1, device=self.device), constraint=constraints.positive
        )
        loc_mu_theta_param = pyro.param("loc_mu_theta", torch.tensor(0.0, device=self.device))
        scale_mu_theta_param = pyro.param(
            "scale_mu_theta",
            torch.tensor(1.0e1, device=self.device),
            constraint=constraints.positive,
        )
        loc_mu_a_param = pyro.param("loc_mu_a", torch.tensor(0.0, device=self.device))
        scale_mu_a_param = pyro.param(
            "scale_mu_a", torch.tensor(1.0e1, device=self.device), constraint=constraints.positive
        )
        alpha_b_param = pyro.param("alpha_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        beta_b_param = pyro.param("beta_b", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        alpha_theta_param = pyro.param(
            "alpha_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        beta_theta_param = pyro.param(
            "beta_theta", torch.tensor(1.0, device=self.device), constraint=constraints.positive
        )
        alpha_a_param = pyro.param("alpha_a", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        beta_a_param = pyro.param("beta_a", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        m_theta_param = pyro.param("loc_ability", torch.zeros(self.num_subjects, device=self.device))
        s_theta_param = pyro.param(
            "scale_ability",
            torch.ones(self.num_subjects, device=self.device),
            constraint=constraints.positive,
        )
        m_b_param = pyro.param("loc_diff", torch.zeros(self.num_items, device=self.device))
        s_b_param = pyro.param(
            "scale_diff",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        m_a_param = pyro.param("loc_slope", torch.zeros(self.num_items, device=self.device))
        s_a_param = pyro.param(
            "scale_slope",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )

        # sample statements
        pyro.sample("mu_b", dist.Normal(loc_mu_b_param, scale_mu_b_param))
        pyro.sample("u_b", dist.Gamma(alpha_b_param, beta_b_param))
        pyro.sample("mu_theta", dist.Normal(loc_mu_theta_param, scale_mu_theta_param))
        pyro.sample("u_theta", dist.Gamma(alpha_theta_param, beta_theta_param))
        pyro.sample("mu_a", dist.Normal(loc_mu_a_param, scale_mu_a_param))
        pyro.sample("u_a", dist.Gamma(alpha_a_param, beta_a_param))

        with pyro.plate("thetas", self.num_subjects, device=self.device):
            pyro.sample("theta", dist.Normal(m_theta_param, s_theta_param))
        with pyro.plate("bs", self.num_items, device=self.device):
            pyro.sample("b", dist.Normal(m_b_param, s_b_param))
            pyro.sample("a", dist.Normal(m_a_param, s_a_param))

    def get_model(self):
        return self.model_hierarchical

    def get_guide(self):
        return self.guide_hierarchical

    def export(self):
        return {
            "ability": pyro.param("loc_ability").data.tolist(),
            "diff": pyro.param("loc_diff").data.tolist(),
            "disc": pyro.param("loc_slope").data.tolist(),
        }
