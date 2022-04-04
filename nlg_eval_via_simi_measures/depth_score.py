from __future__ import absolute_import, division, print_function
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.preprocessing import normalize
from sklearn.covariance import MinCovDet as MCD
from sklearn.decomposition import PCA
import logging

import ot
import geomloss

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)


class DepthScoreMetric:
    def __init__(self, model_name="bert-base-uncased", layers_to_consider=9, considered_measure='irw', p=None, eps=None,
                 n_alpha=None):

        """
        DepthScore metric
        :param model_name: model name or path from HuggingFace Librairy
        :param layers_to_consider: layer to use in the pretrained model
        :param considered_measure: measure of similarity to use should be in ["irw", "ai_irw", "wasserstein", "sliced", "mmd"]
        :param p: the power of the ground cost.
        :param eps:   the highest level set.
        :param n_alpha: The Monte-Carlo parameter for the approximation of the integral
        over alpha.
        """
        self.n_alpha = 5 if n_alpha is None else n_alpha
        self.eps = 0.3 if eps is None else eps
        self.p = 5 if p is None else p
        self.model_name = model_name
        self.load_tokenizer_and_model()
        self.considered_measure = considered_measure
        assert considered_measure in ["irw", "ai_irw", "wasserstein", "sliced", "mmd"]
        self.layers_to_consider = layers_to_consider
        assert layers_to_consider < self.model.config.num_hidden_layers + 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_tokenizer_and_model(self):
        """
        Loading and initializing the chosen model and tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained('{}'.format(self.model_name))
        model = AutoModelForMaskedLM.from_pretrained('{}'.format(self.model_name))
        model.config.output_hidden_states = True
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def evaluate_batch(self, batch_hyps, batch_refs, idf_hyps=None, idf_ref=None):
        """
        :param batch_hyps: hypothesis list of string sentences
        :param batch_refs: reference list of string sentences
        :return: dictionnary of scores
        """
        ###############################################
        ## Extract Embeddings From Pretrained Models ##
        ###############################################
        if isinstance(batch_hyps, str):
            batch_hyps = [batch_hyps]
        if isinstance(batch_refs, str):
            batch_refs = [batch_refs]
        nb_sentences = len(batch_refs)
        depth_scores = []
        assert len(batch_hyps) == len(batch_refs)

        model = self.model.to(self.device)

        with torch.no_grad():
            ###############################################
            ## Extract Embeddings From Pretrained Models ##
            ###############################################
            batch_refs = self.tokenizer(batch_refs, return_tensors='pt', padding=True, truncation=True).to(self.device)
            batch_refs_embeddings_ = model(**batch_refs)[-1]

            batch_hyps = self.tokenizer(batch_hyps, return_tensors='pt', padding=True, truncation=True).to(self.device)
            batch_hyps_embeddings_ = model(**batch_hyps)[-1]

            batch_refs_embeddings = [batch_refs_embeddings_[i] for i in [self.layers_to_consider]]
            batch_hyps_embeddings = [batch_hyps_embeddings_[i] for i in [self.layers_to_consider]]

            batch_refs_embeddings = torch.cat([i.unsqueeze(0) for i in batch_refs_embeddings])
            batch_refs_embeddings.div_(torch.norm(batch_refs_embeddings, dim=-1).unsqueeze(-1))
            batch_hyps_embeddings = torch.cat([i.unsqueeze(0) for i in batch_hyps_embeddings])
            batch_hyps_embeddings.div_(torch.norm(batch_hyps_embeddings, dim=-1).unsqueeze(-1))

            ref_tokens_id = batch_refs['input_ids'].cpu().tolist()
            hyp_tokens_id = batch_hyps['input_ids'].cpu().tolist()

            ######################################
            ## Unbatched Depth Score Prediction ##
            ######################################
            for index_sentence in tqdm(range(nb_sentences), 'Depth Score Progress'):
                ref_tokens = [i for i in self.tokenizer.convert_ids_to_tokens(ref_tokens_id[index_sentence],
                                                                              skip_special_tokens=False) if
                              i != self.tokenizer.pad_token]
                hyp_tokens = [i for i in self.tokenizer.convert_ids_to_tokens(hyp_tokens_id[index_sentence],
                                                                              skip_special_tokens=False) if
                              i != self.tokenizer.pad_token]

                ref_ids = [k for k, w in enumerate(ref_tokens) if True]
                hyp_ids = [k for k, w in enumerate(hyp_tokens) if True]

                ref_embedding_i = batch_refs_embeddings[:, index_sentence, ref_ids, :]
                hyp_embedding_i = batch_hyps_embeddings[:, index_sentence, hyp_ids, :]
                measures_locations_ref = ref_embedding_i.permute(1, 0, 2).cpu().numpy().tolist()
                measures_locations_ref = [np.array(i) for i in measures_locations_ref]
                measures_locations_hyps = hyp_embedding_i.permute(1, 0, 2).cpu().numpy().tolist()
                measures_locations_hyps = [np.array(i) for i in measures_locations_hyps]

                dict_score = self.depth_score(measures_locations_ref, measures_locations_hyps)
                depth_scores.append(dict_score)
        depth_scores_dic = {}
        for k in dict_score.keys():
            depth_scores_dic[k] = []
            for score in depth_scores:
                depth_scores_dic[k].append(score[k])
        return depth_scores_dic

    def prepare_idfs(self, hyps, refs):
        """
        Depth Score does not use idfs
        """
        return None, None

    def depth_score(self, measures_locations_ref, measures_locations_hyps):
        """
        :param measures_locations_ref: discrete input measures of one reference
        :param measures_locations_hyps: discrete input measures of one hypothesis
        :return:
        """
        ##################################################################
        ## Compute Score between the location and hypothesis reference ##
        ##################################################################
        measures_locations_ref = np.array(measures_locations_ref).squeeze(1)
        measures_locations_hyps = np.array(measures_locations_hyps).squeeze(1)
        depth_score = dr_distance(measures_locations_ref, measures_locations_hyps, n_alpha=self.n_alpha,
                                  n_dirs=10000, data_depth=self.considered_measure, eps_min=self.eps, eps_max=1,
                                  p=self.p)
        return {'depth_score': depth_score}

    @property
    def supports_multi_ref(self):
        """
        :return: BaryScore does not support multi ref
        """
        return False


########################################################
#################### Sampled distribution ########################
########################################################


########################################################
#################### Some useful functions ########################
########################################################


def cov_matrix(X, robust=False):
    """
    :param X: input matrix
    :param robust: if true compute a robust estimate
    :return: covariance matrix of X
    """

    if robust:
        cov = MCD().fit(X)
        sigma = cov.covariance_
    else:
        sigma = np.cov(X.T)

    return sigma


def standardize(X, robust=False):
    """
    :param X:  input matrix
    :param robust: if true compute a robust estimate of the covariance matrix
    :return: square inverse f the covariance matrix of X.
    """

    sigma = cov_matrix(X, robust)
    n_samples, n_features = X.shape
    rank = np.linalg.matrix_rank(X)

    if (rank < n_features):
        pca = PCA(rank)
        pca.fit(X)
        X_transf = pca.fit_transform(X)
        sigma = cov_matrix(X_transf)
    else:
        X_transf = X.copy()

    u, s, _ = np.linalg.svd(sigma)
    square_inv_matrix = u / np.sqrt(s)

    return X_transf @ square_inv_matrix


########################################################
#################### Sampled distributions ########################
########################################################

def sampled_sphere(n_dirs, d):
    """
    :param n_dirs: number of direction to consider
    :param d: dimension of the unite sphere
    :return: ndirs samples of d-dimensional uniform distribution on the
        unit sphere
    """

    mean = np.zeros(d)
    identity = np.identity(d)
    U = np.random.multivariate_normal(mean=mean, cov=identity, size=n_dirs)

    return normalize(U)


def Wasserstein(X, Y):
    """
    :param X: input distribution X
    :param Y: input distribution Y
    :return: wasserstein distance between X and Y
    """
    M = ot.dist(X, Y)
    n = len(X)
    m = len(Y)
    w_X = np.zeros(n) + 1 / n
    w_Y = np.zeros(m) + 1 / m

    return ot.emd2(w_X, w_Y, M)


def SW(X, Y, ndirs, p=2, max_sliced=False):
    """
    :param X: input distribution X
    :param Y: input distribution Y
    :param ndirs: number of direction to consider when slicing
    :param p: order of the Sliced wasserstein distance
    :param max_sliced: if true take the maximum, if false the mean is applied
    :return: Sliced-Wasserstein distance between X and Y
    """
    n, d = X.shape
    U = sampled_sphere(ndirs, d)
    Z = np.matmul(X, U.T)
    Z2 = np.matmul(Y, U.T)
    Sliced = np.zeros(ndirs)
    for k in range(ndirs):
        Sliced[k] = ot.emd2_1d(Z[:, k], Z2[:, k], p=2)
    if (max_sliced == True):
        return (np.max(Sliced)) ** (1 / p)
    else:
        return (np.mean(Sliced)) ** (1 / p)


def MMD(X, Y):
    """
    :param X: input distribution X
    :param Y: input distribution Y
    :return:  MMD cost between X and Y
    """
    return geomloss.SamplesLoss("gaussian")(torch.tensor(X), torch.tensor(Y)).item()


########################################################
#################### Data Depths ########################
########################################################

def ai_irw(X, AI=True, robust=False, n_dirs=None, random_state=None):
    """
    :param X: Array of shape (n_samples, n_features)
            The training set.
    :param AI: bool
        if True, the affine-invariant version of irw is computed.
        If False, the original irw is computed.
    :param robust:  if robust is true, the MCD estimator of the covariance matrix
        is performed.
    :param n_dirs:   The number of random directions needed to approximate
        the integral over the unit sphere.
        If None, n_dirs is set as 100* n_features.
    :param random_state:  The random state.

    :return:   Depth score of each element in X_test, where the considered depth is (Affine-invariant-) integrated rank
        weighted depth of X_test w.r.t. X
    """

    if random_state is None:
        random_state = 0

    np.random.seed(random_state)

    if AI:
        X_reduced = standardize(X, robust)
    else:
        X_reduced = X.copy()

    n_samples, n_features = X_reduced.shape

    if n_dirs is None:
        n_dirs = n_features * 100

    # Simulated random directions on the unit sphere.
    U = sampled_sphere(n_dirs, n_features)

    sequence = np.arange(1, n_samples + 1)
    depth = np.zeros((n_samples, n_dirs))

    proj = np.matmul(X_reduced, U.T)
    rank_matrix = np.matrix.argsort(proj, axis=0)

    for k in range(n_dirs):
        depth[rank_matrix[:, k], k] = sequence

    depth = depth / (n_samples * 1.)
    depth_score = np.minimum(depth, 1 - depth)
    ai_irw_score = np.mean(depth_score, axis=1)

    return ai_irw_score


import numpy as np


def dr_distance(X, Y, n_alpha=10, n_dirs=100, data_depth='tukey', eps_min=0,
                eps_max=1, p=2, random_state=None):
    """
    :param X: array of shape (n_samples, n_features)
        The first sample.
    :param Y: array of shape (n_samples, n_features)
        The second sample.
    :param n_alpha: The Monte-Carlo parameter for the approximation of the integral
        over alpha.
    :param n_dirs: The number of directions for approximating the supremum over
        the unit sphere.
    :param data_depth: depth to consider in  {'tukey', 'projection', 'irw', 'ai_irw'}
    :param eps_min: float in [0,eps_max]
        the lowest level set.
    :param eps_max: float in [eps_min,1]
        the highest level set.
    :param p:    the power of the ground cost.
    :param random_state:  The random state.
    :return: the computed pseudo-metric score.
    """

    if random_state is None:
        random_state = 0

    np.random.seed(random_state)

    if data_depth not in {'tukey', 'projection', 'irw', 'ai_irw', 'wasserstein', 'mmd', 'sliced'}:
        raise NotImplementedError('This data depth is not implemented')

    if eps_min > eps_max:
        raise ValueError('eps_min must be lower than eps_max')

    if eps_min < 0 or eps_min > 1:
        raise ValueError('eps_min must be in [0,eps_max]')

    if eps_max < 0 or eps_max > 1:
        raise ValueError('eps_min must be in [eps_min,1]')

    _, n_features = X.shape
    if data_depth == "irw":
        depth_X = ai_irw(X, AI=False, n_dirs=n_dirs)
        depth_Y = ai_irw(Y, AI=False, n_dirs=n_dirs)
    elif data_depth == "ai_irw":
        depth_X = ai_irw(X, AI=True, n_dirs=n_dirs)
        depth_Y = ai_irw(Y, AI=True, n_dirs=n_dirs)
    elif data_depth == 'wasserstein':
        return Wasserstein(X, Y)
    elif data_depth == 'sliced':
        return SW(X, Y, ndirs=10000)
    elif data_depth == 'mmd':
        return MMD(X, Y)

        # draw n_dirs vectors of the unit sphere in dimension n_features.
    U = sampled_sphere(n_dirs, n_features)
    proj_X = np.matmul(X, U.T)
    proj_Y = np.matmul(Y, U.T)

    liste_alpha = np.linspace(int(eps_min * 100), int(eps_max * 100), n_alpha)
    quantiles_DX = [np.percentile(depth_X, j) for j in liste_alpha]
    quantiles_DY = [np.percentile(depth_Y, j) for j in liste_alpha]

    dr_score = 0
    for i in range(n_alpha):
        d_alpha_X = np.where(depth_X >= quantiles_DX[i])[0]
        d_alpha_Y = np.where(depth_Y >= quantiles_DY[i])[0]
        supp_X = np.max(proj_X[d_alpha_X], axis=0)
        supp_Y = np.max(proj_Y[d_alpha_Y], axis=0)
        dr_score += np.max((supp_X - supp_Y) ** p)

    return (dr_score / n_alpha) ** (1 / p)


if __name__ == '__main__':
    model_name = 'distilbert-base-uncased'  # we consider distillbert for speed concerns
    metric_call = DepthScoreMetric(model_name, layers_to_consider=4)

    ref = [
        'I like my cakes very much', 'I hate these cakes so much']
    hypothesis = ['I like my cakes very much', 'I like my cakes very much']

    final_preds = metric_call.evaluate_batch(ref, hypothesis)
    print(final_preds)
