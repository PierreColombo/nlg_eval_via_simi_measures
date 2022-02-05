from __future__ import absolute_import, division, print_function

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import torch
import os
from math import log
from collections import defaultdict, Counter


class InfoLM:
    def __init__(self, model_name="bert-base-uncased", temperature=0.25, measure_to_use='fisher_rao',
                 use_idf_weights=True, alpha=None, beta=None):
        """
        :param model_name: model name or path from HuggingFace Librairy
        :param temperature: temperature to calibrate the LM
        :param measure_to_use: which measure of information to use
        :param use_idf_weights: if true use tf-idf weights
        :param alpha: alpha parameter in the ab, alpha or renyi div
        :param beta: beta parameter in the ab or beta div
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_tokenizer_and_model()
        self.use_idf_weights = use_idf_weights
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.measure_to_use = measure_to_use
        assert self.measure_to_use in ['kl', 'alpha', 'renyi', 'beta', 'ab', 'l1', "l2", "linf", 'fisher_rao']
        if self.measure_to_use in ['alpha', 'ab', 'renyi']: assert alpha is not None
        if self.measure_to_use in ['beta', 'ab']:  assert beta is not None

    def prepare_idfs(self, hyps, refs):
        """
        :param hyps: hypothesis list of string sentences has to be computed at corpus level
        :param refs:reference list of string sentences has to be computed at corpus level
        """
        t_hyps = self.tokenizer(hyps)['input_ids']
        t_refs = self.tokenizer(refs)['input_ids']
        idf_dict_ref = self.ref_list_to_idf(t_refs)
        idf_dict_hyp = self.ref_list_to_idf(t_hyps)
        self.idf_dict_hyp, self.idf_dict_ref = idf_dict_hyp, idf_dict_ref
        return idf_dict_hyp, idf_dict_ref

    def ref_list_to_idf(self, input_refs):
        """
        :param input_refs: list of input reference
        :return: idf dictionnary
        """
        idf_count = Counter()
        num_docs = len(input_refs)

        idf_count.update(sum([list(set(i)) for i in input_refs], []))

        idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
        idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
        return idf_dict

    def load_tokenizer_and_model(self):
        """
        Loading and initializing the chosen model and tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained('{}'.format(self.model_name))
        model = AutoModelForMaskedLM.from_pretrained('{}'.format(self.model_name))
        model.config.output_hidden_states = True
        model.eval()
        self.tokenizer = tokenizer
        self.model = model.to(self.device)

    def _safe_divide(self, numerator, denominator):
        """
        :param numerator: quotient numerator
        :param denominator: quotient denominator
        :return: safe divide of numerator/denominator
        """
        return numerator / (denominator + 1e-30)

    def nan_to_num(self, tensor):
        """
        :param tensor: input tensor
        :return: tensor without nan
        """
        tensor[tensor != tensor] = 0
        return tensor

    def alpha_div(self, ref_dist, hypo_dist):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: alpha divergence between the reference and hypothesis distribution
        """
        alpha = self.alpha
        assert alpha != 1 and alpha != 0
        return 1 / (alpha * (alpha - 1)) - torch.sum(ref_dist ** alpha * hypo_dist ** (1 - alpha), dim=-1) / (
                alpha * (alpha - 1))

    def linfinity(self, ref_dist, hypo_dist):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: l infinity norm between the reference and hypothesis distribution
        """
        return torch.norm(ref_dist - hypo_dist, p=float("inf"), dim=-1)

    def l2(self, ref_dist, hypo_dist):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: l2 norm between the reference and hypothesis distribution
        """
        return torch.norm(ref_dist - hypo_dist, p=2, dim=-1)

    def l1(self, ref_dist, hypo_dist):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: l1 norm between the reference and hypothesis distribution
        """
        return torch.norm(ref_dist - hypo_dist, p=1, dim=-1)

    def fisher_rao(self, ref_dist, hypo_dist):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: fisher rao distance between the reference and hypothesis distribution
        """
        rao_fi = torch.clamp(
            torch.sum(torch.sqrt(ref_dist) * torch.sqrt(
                hypo_dist),
                      dim=-1), 0, 1)
        rao = 2 * torch.acos(rao_fi)
        return rao

    def kl_div(self, ref_dist, hypo_dist):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :return: kl divergence between the reference and hypothesis distribution
        """
        kl = torch.sum(ref_dist * torch.log(hypo_dist / ref_dist), dim=-1)
        return kl

    def renyi_div(self, ref_dist, hypo_dist):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :return: renyi divergence between the reference and hypothesis distribution
        """
        alpha = self.alpha
        assert alpha != 1
        return torch.log(torch.sum(ref_dist ** alpha * hypo_dist ** (1 - alpha), dim=-1)) / (alpha - 1)

    def beta_div(self, ref_dist, hypo_dist):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :param beta: beta parameter of the divergence
        :return: beta divergence between the reference and hypothesis distribution
        """
        beta = self.beta
        assert beta != -1
        assert beta != 0
        first_term = torch.log(torch.sum(ref_dist ** (beta + 1), dim=-1)) / (beta * (beta + 1))
        second_term = torch.log(torch.sum(hypo_dist ** (beta + 1), dim=-1)) / (beta + 1)
        third_term = torch.log(torch.sum(ref_dist * hypo_dist ** (beta), dim=-1)) / (beta)
        return first_term + second_term - third_term

    def ab_div(self, ref_dist, hypo_dist):
        """
        :param ref_dist: discreate input reference distribution over the vocabulary
        :param hypo_dist: discreate hypothesis reference distribution over the vocabulary
        :param alpha: alpha parameter of the divergence
        :param beta: beta parameter of the divergence
        :return: ab divergence between the reference and hypothesis distribution
        """
        beta = self.beta
        alpha = self.alpha
        assert alpha != 0
        assert beta != 0
        assert beta + alpha != 0
        first_term = torch.log(torch.sum(ref_dist ** (beta + alpha), dim=-1)) / (beta * (beta + alpha))
        second_term = torch.log(torch.sum(hypo_dist ** (beta + alpha), dim=-1)) / (alpha * (beta + alpha))
        third_term = torch.log(torch.sum((ref_dist ** (alpha)) * (hypo_dist ** (beta)), dim=-1)) / (beta * alpha)
        return first_term + second_term - third_term

    def compute_infolm(self, ref_distribution, hyp_distribution):
        """
        :param ref_distribution: aggregated reference distribution (weighted or not / calibrated of not)
        :param hyp_distribution: : aggregated hypothesis distribution (weighted or not  / calibrated of not)
        :return: infoLM score
        """
        if self.measure_to_use == 'kl':
            measure = self.kl_div
        elif self.measure_to_use == 'alpha':
            measure = self.alpha_div
        elif self.measure_to_use == 'renyi':
            measure = self.renyi_div
        elif self.measure_to_use == 'beta':
            measure = self.beta_div
        elif self.measure_to_use == 'ab':
            measure = self.ab_div
        elif self.measure_to_use == 'l1':
            measure = self.l1
        elif self.measure_to_use == 'l2':
            measure = self.l2
        elif self.measure_to_use == 'linf':
            measure = self.linfinity
        elif self.measure_to_use == 'fisher_rao':
            measure = self.fisher_rao
        else:
            raise NotImplementedError
        normal_div = self.nan_to_num(measure(ref_distribution, hyp_distribution))
        reversed_div = self.nan_to_num(measure(hyp_distribution, ref_distribution))
        return {
            "{}".format(self.measure_to_use): normal_div.tolist(),
            "r_{}".format(self.measure_to_use): reversed_div.tolist(),
            "sim_{}".format(self.measure_to_use): ((normal_div + reversed_div) / 2).tolist(),
        }

    def get_distribution(self, tokenizer_output, idf_dic):
        """
        :param tokenizer_output:
        :param idf_dic:
        :return:
        """
        final_distribution = []
        idfs = []
        max_length = self.tokenizer(tokenizer_output, return_tensors="pt", padding=True, truncation=True).to(
            self.device)['input_ids'].size()[-1]
        for index_to_mask in range(max_length):
            unmasked_data = self.tokenizer(tokenizer_output, return_tensors="pt", padding=True, truncation=True).to(
                self.device)
            if self.use_idf_weights:
                ids_masked_list = unmasked_data['input_ids'][:, index_to_mask].tolist()
                tf_idf_term = torch.tensor([idf_dic[id] for id in ids_masked_list]).unsqueeze(-1)
                idfs.append(tf_idf_term)
            labels = unmasked_data['input_ids'].clone()
            masked_indices = torch.zeros_like(labels).to(self.device).bool()
            masked_indices[:, index_to_mask] = 1
            labels[~masked_indices] = -100
            masked_input_ids = unmasked_data['input_ids']
            masked_input_ids[:, index_to_mask] = self.tokenizer.mask_token_id
            unmasked_data['input_ids'] = masked_input_ids
            outputs = self.model(**unmasked_data, labels=labels)
            logits_distribution = outputs[1][:, index_to_mask, :].cpu()
            dict_logits_distribution = {}
            pad_token_mask = ((labels.eq(self.tokenizer.pad_token_id)[:, index_to_mask] |
                               labels.eq(self.tokenizer.cls_token_id)[:,
                               index_to_mask]) |
                              labels.eq(self.tokenizer.sep_token_id)[:, index_to_mask])
            pad_token_mask = pad_token_mask.unsqueeze(1).repeat(1, logits_distribution.size(-1))

            dict_logits_distribution[str(self.temperature)] = torch.nn.Softmax()(
                logits_distribution / self.temperature)
            if self.use_idf_weights:
                dict_logits_distribution[str(self.temperature)] = dict_logits_distribution[
                                                                      str(self.temperature)] * tf_idf_term

            dict_logits_distribution[str(self.temperature)][pad_token_mask] = torch.ones_like(
                dict_logits_distribution[str(self.temperature)][pad_token_mask]) * 10000
            del masked_input_ids
            del labels
            del unmasked_data
            del outputs
            final_distribution.append(dict_logits_distribution)
        return final_distribution, idfs

    def evaluate_batch(self, batch_hyps, batch_refs, idf_hyps=None, idf_ref=None):
        """
        :param batch_hyps: hypothesis list of string sentences
        :param batch_refs: reference list of string sentences
        :param idf_hyps: idfs of hypothesis computed at corpus level
        :param idf_ref: idfs of references computed at corpus level
        :return: dictionary of scores
        """
        if self.use_idf_weights:
            if (idf_hyps is None) and (idf_ref is None):
                idf_hyps, idf_ref = self.idf_dict_hyp, self.idf_dict_ref
            idf_hyps[self.model.config.pad_token_id] = 0  # for padding
            idf_ref[self.model.config.pad_token_id] = 0
        with torch.no_grad():
            dict_final_distribution_batch_refs, idfs_ref = self.get_distribution(batch_refs,
                                                                                 idf_ref if self.use_idf_weights else None)
            dict_final_distribution_batch_hypothesis, idfs_hyp = self.get_distribution(batch_hyps,
                                                                                       idf_hyps if self.use_idf_weights else None)
        mask_ref = self.tokenizer(batch_refs, return_tensors="pt", padding=True, truncation=True)['input_ids']
        mask_hyps = self.tokenizer(batch_hyps, return_tensors="pt", padding=True, truncation=True)['input_ids']
        mask_ref = ((mask_ref.eq(self.tokenizer.sep_token_id) |
                     mask_ref.eq(self.tokenizer.cls_token_id)) |
                    mask_ref.eq(self.tokenizer.pad_token_id))
        mask_hyps = ((mask_hyps.eq(self.tokenizer.sep_token_id) |
                      mask_hyps.eq(self.tokenizer.cls_token_id)) |
                     mask_hyps.eq(self.tokenizer.pad_token_id))

        mask_words_hyps = torch.sum(~mask_hyps, dim=1)
        mask_words_refs = torch.sum(~mask_ref, dim=1)
        mask_ref = mask_ref.unsqueeze(-1).repeat(1, 1,
                                                 dict_final_distribution_batch_hypothesis[0][
                                                     str(self.temperature)].size(
                                                     -1))
        mask_hyps = mask_hyps.unsqueeze(-1).repeat(1, 1,
                                                   dict_final_distribution_batch_hypothesis[0][
                                                       str(self.temperature)].size(
                                                       -1))

        final_distribution_batch_refs = torch.cat(
            [i[str(self.temperature)].unsqueeze(1) for i in dict_final_distribution_batch_refs],
            dim=1)
        final_distribution_batch_refs[mask_ref] = 0
        final_distribution_batch_hypothesis = torch.cat(
            [i[str(self.temperature)].unsqueeze(1) for i in dict_final_distribution_batch_hypothesis], dim=1)
        final_distribution_batch_hypothesis[mask_hyps] = 0
        if self.use_idf_weights:
            sum_distribution_refs = torch.sum(final_distribution_batch_refs, dim=1) / torch.sum(
                torch.cat(idfs_ref, dim=-1),
                dim=-1).unsqueeze(-1)
            sum_distribution_hypothesis = torch.sum(final_distribution_batch_hypothesis,
                                                    dim=1) / torch.sum(torch.cat(idfs_hyp, dim=-1),
                                                                       dim=-1).unsqueeze(-1)
        else:
            sum_distribution_hypothesis = torch.sum(final_distribution_batch_hypothesis,
                                                    dim=1) / mask_words_hyps.unsqueeze(-1).repeat(1,
                                                                                                  final_distribution_batch_hypothesis[
                                                                                                      0].size(
                                                                                                      -1))
            sum_distribution_refs = torch.sum(final_distribution_batch_refs, dim=1) / mask_words_refs.unsqueeze(
                -1).repeat(1, final_distribution_batch_hypothesis[0].size(-1))

        info_dic = self.compute_infolm(sum_distribution_hypothesis, sum_distribution_refs)
        return info_dic


if __name__ == '__main__':

    for measure in ['kl', 'alpha', 'renyi', 'beta', 'ab', 'l1', "l2", "linf", 'fisher_rao']:
        metric = InfoLM(measure_to_use=measure, alpha=0.25, beta=0.25, temperature=1, use_idf_weights=False)

        ref = ['I like my cakes very much', 'I like my cakes very much']
        hypothesis = ['I like my cakes very much', 'I hate these cakes very much']

        idf_ref, idf_hypot = metric.prepare_idfs(ref, hypothesis)

        final_preds = metric.evaluate_batch(ref, hypothesis)
        print(final_preds)
