###
# s-SuStaIn: a scaled implementation of pySuStaIn with major modifications
#
# Original authors: Peter Wijeratne (p.wijeratne@ucl.ac.uk) and Leon Aksman (leon.aksman@loni.usc.edu)
# Contributors: Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk), Cameron Shand (c.shand@ucl.ac.uk)
# Significant modifications by:
#    - Raghav Tandon – algorithmic extensions and structural changes
#    - Neel Sarkar – additional refinements and integration work
#
# This implementation builds upon the original pySuStaIn framework and introduces enhancements
# for scalability and integration, developed as part of the ReduXis project.
#
# For code reference, please see the following repositories:
# 1. The original pySuStaIn framework: https://github.com/ucl-pond/pySuStaIn
# 2. The ReduXis project: https://github.com/pathology-dynamics/ReduXis
#
# If you use s-SuStaIn, please cite the following core papers:
# 1. The original SuStaIn paper:    https://doi.org/10.1038/s41467-018-05892-0
# 2. The pySuStaIn software paper:  https://doi.org/10.1016/j.softx.2021.100811
# 3. The s-SuStaIn software paper:  https://pmc.ncbi.nlm.nih.gov/articles/PMC11881980
#
# Please also cite the corresponding progression pattern model you use:
# 1. The piece-wise linear z-score model (i.e. ZscoreSustain):  https://doi.org/10.1038/s41467-018-05892-0
# 2. The event-based model (i.e. MixtureSustain):               https://doi.org/10.1016/j.neuroimage.2012.01.062
#    with Gaussian mixture modeling (i.e. 'mixture_gmm'):       https://doi.org/10.1093/brain/awu176
#    or kernel density estimation (i.e. 'mixture_kde'):         https://doi.org/10.1002/alz.12083
# 3. The model for discrete ordinal data (i.e. OrdinalSustain): https://doi.org/10.3389/frai.2021.613261
#
# Thank you for supporting the SuStaIn ecosystem and its continued development.
###

import pdb
import os
from pathlib import Path
import pickle
import csv
import os
import multiprocessing
from functools import partial, partialmethod

import pathos
import warnings
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
from scipy.special import logsumexp
import time

from sSuStaIn.AbstractSustain import AbstractSustainData
from sSuStaIn.AbstractSustain import AbstractSustain

#*******************************************
#The data structure class for MixtureSustain. It holds the positive/negative likelihoods that get passed around and re-indexed in places.
class sEBMSustainData(AbstractSustainData):

    def __init__(self, L_yes, L_no, n_stages):

        assert(L_yes.shape[0] == L_no.shape[0] and L_yes.shape[1] == L_no.shape[1])

        self.L_yes          = L_yes
        self.L_no           = L_no
        self.n_stages       = n_stages
        self.L_yes_log      = np.log(L_yes)
        self.L_no_log       = np.log(L_no)

    def getNumSamples(self):
        return self.L_yes.shape[0]

    def getNumBiomarkers(self):
        return self.L_no.shape[1]

    def getNumStages(self):
        return self.n_stages

    def reindex(self, index):
        return sEBMSustainData(self.L_yes[index,], self.L_no[index,], self.n_stages)

#*******************************************
#An implementation of the AbstractSustain class with mixture model based events
class sEBMSustain(AbstractSustain):

    def __init__(self,
                 L_yes,
                 L_no,
                 n_stages, 
                 stage_size_init, 
                 min_stage_size,
                 p_absorb,
                 rep_opt,
                 biomarker_labels,
                 N_startpoints,
                 N_S_max,
                 N_iterations_MCMC_init,
                 N_iterations_MCMC,
                 N_em,
                 output_folder,
                 dataset_name,
                 use_parallel_startpoints,
                 seed=None):
        # The initializer for the mixture model based events implementation of AbstractSustain
        # Parameters:
        #   L_yes                       - probability of positive class for all subjects across all biomarkers (from mixture modelling)
        #                                 dim: number of subjects x number of biomarkers
        #   L_no                        - probability of negative class for all subjects across all biomarkers (from mixture modelling)
        #                                 dim: number of subjects x number of biomarkers
        #   biomarker_labels            - the names of the biomarkers as a list of strings
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop
        #   seed                        - random number seed

        N                               =  L_yes.shape[1] # number of biomarkers
        assert (len(biomarker_labels) == N), "number of labels should match number of biomarkers"

        self.biomarker_labels           = biomarker_labels
        self.n_stages                   = n_stages
        self.__sustainData              = sEBMSustainData(L_yes, L_no, self.n_stages)
        self.stage_size_init            = stage_size_init
        self.min_stage_size             = min_stage_size
        self.p_absorb                   = p_absorb
        self.rep_opt                    = rep_opt
        self.N_iterations_MCMC_init     = N_iterations_MCMC_init
        self.N_em                       = N_em
        assert self.n_stages == len(stage_size_init), "number of stages should match with the number of elements in stage_size_init"
        assert min(self.stage_size_init) >= self.min_stage_size, "no stage should have fewer biomarkers than what are required by min_stage_size"
        assert self.p_absorb < 1 and self.p_absorb >= 0, "the probability should be less than 1, but can include 0"


        super().__init__(self.__sustainData,
                         N_startpoints,
                         N_S_max,
                         N_iterations_MCMC,
                         output_folder,
                         dataset_name,
                         use_parallel_startpoints,
                         seed)

    # NEW METHODS IN sEBMSustain
    def _initialise_sequence(self, sustainData, rng):
        num_biomarkers = sustainData.getNumBiomarkers()

        # Validation
        assert sum(self.stage_size_init) == num_biomarkers, (
            f"Sum of stage_size_init ({sum(self.stage_size_init)}) "
            f"must equal total biomarkers ({num_biomarkers})"
        )
        assert all(s >= self.min_stage_size for s in self.stage_size_init), (
            f"Every stage_size_init value must be >= min_stage_size ({self.min_stage_size})"
        )

        # Permute and slice into stages
        permuted = rng.permutation(num_biomarkers).astype(int)
        S_dict = self._dictionarize_sequence(permuted, self.stage_size_init)

        # Return as a single-element list for consistency
        return [S_dict]


    def _dictionarize_sequence(self, S, stage_size):
        stages_cumsum = np.cumsum(stage_size, dtype=int)
        S_dict = {}
        prev_idx = 0
        for stage_idx, end_idx in enumerate(stages_cumsum):
            # Slice out exactly stage_size[stage_idx] biomarkers
            S_dict[stage_idx] = S[prev_idx:end_idx]
            prev_idx = end_idx
        return S_dict


    def _get_shape(self, S_dict):
        assert isinstance(S_dict, dict), "Sequence must be a dict"
        N_stages = len(S_dict)
        assert N_stages == self.n_stages, f"Expected {self.n_stages} stages, got {N_stages}"
        shape = [len(S_dict[i]) for i in range(N_stages)]
        assert min(shape) >= self.min_stage_size, (
            f"Each stage should have ≥ {self.min_stage_size} biomarkers, "
            f"but got stage sizes {shape}"
        )
        return shape

    def _flatten_sequence(self, S):
        flatten_S = np.vstack([self._flatten_S_dict(s) for s in S])
        return flatten_S


    def _flatten_S_dict(self, S_dict):
        # S_dict is dictionary, NOT a list of dictionaries
        flatten_S = []
        stages = len(S_dict)
        for k in range(stages):
            flatten_S.append(S_dict[k])
        return np.hstack(flatten_S)

    # INHERITED METHODS FROM AbstractSustain
    def _calculate_likelihood(self, sustainData, S, f):
        # Computes the likelihood of a mixture of models
        #
        #
        # OUTPUTS:
        # loglike               - the log-likelihood of the current model
        # total_prob_subj       - the total probability of the current SuStaIn model for each subject
        # total_prob_stage      - the total probability of each stage in the current SuStaIn model
        # total_prob_cluster    - the total probability of each subtype in the current SuStaIn model
        # p_perm_k              - the probability of each subjects data at each stage of each subtype in the current SuStaIn model

        M                                   = sustainData.getNumSamples()
        N_S                                 = len(S)
        N                                   = sustainData.getNumStages()

        f                                   = np.array(f).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        p_perm_k_log                            = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            shape_S = self._get_shape(S[s])
            p_perm_k_log[:, :, s]               = self._calculate_likelihood_stage(sustainData, S[s], shape_S)

        p_perm_k_weighted_log                   = p_perm_k_log + np.log(f_val_mat)
        total_prob_cluster_log              = logsumexp(p_perm_k_weighted_log, axis=1)
        total_prob_stage_log                = logsumexp(p_perm_k_weighted_log, axis=2)
        total_prob_subj_log                 = logsumexp(total_prob_stage_log, axis=1)
        loglike                             = np.sum(total_prob_subj_log)
        return loglike, total_prob_subj_log, total_prob_stage_log, total_prob_cluster_log, p_perm_k_log

    def _calculate_likelihood_stage(self, sustainData, S, stage_size):
        '''
        S - Should be a dictionary
        Computes the likelihood of a single event based model
        stage_size - gives the shape of S (number of biomarkers in each cluster)

        Inputs:
        =======
        sustainData - a MixtureData type that contains:
            L_yes - likelihood an event has occurred in each subject
                    dim: number of subjects x number of biomarkers
            L_no -  likelihood an event has not occurred in each subject
                    dim: number of subjects x number of biomarkers
            S -     the current (dict) ordering for a particular subtype
                    dim: 1 x number of events
        Outputs:
        ========
         p_perm_k - the probability of each subjects data at each stage of a particular subtype
         in the SuStaIn model
        '''

        M = sustainData.getNumSamples()
        N = sustainData.getNumStages()
        N_b = sustainData.getNumBiomarkers()
        ss = self._get_shape(S)
        assert ss == stage_size, "passed stage shape should correspond to the dictionary shape"
        S = self._flatten_S_dict(S) # Flatten the dictionary form of S
        assert len(ss) == N, "the number of biomarker clusters should match the number of stages"
        assert sum(ss) == N_b, "sum of cluster sizes should be equal to total number of biomarkers"
        sample_idx = np.cumsum(ss[:-1])
        S_int = S.astype(int)
        arange_Np1 = np.arange(0, N+1) # redundant (leaving due to legacy)
        p_perm_k_log = np.zeros((M, N+1))

        #**** THIS VERSION IS ROUGHLY 10x FASTER THAN THE ONE BELOW
        cp_yes = np.cumsum(sustainData.L_yes_log[:, S_int], 1)
        cp_no = np.cumsum(sustainData.L_no_log[:,  S_int[::-1]],  1)   #do the cumulative product from the end of the sequence

        # Even faster version to avoid loops
        p_perm_k_log[:, 0] = cp_no[:, -1]
        p_perm_k_log[:, -1] = cp_yes[:, -1]
        p_perm_k_log[:, 1:-1] =  cp_yes[:, :-1][:,sample_idx - 1] + cp_no[:, :-1][:,int(N_b) - sample_idx - 1]

        p_perm_k_log += np.log(1 / (N + 1))

        return p_perm_k_log


    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        # Optimise the parameters of the SuStaIn model

        M                                   = sustainData.getNumSamples()
        N_S                                 = len(S_init)
        N                                   = sustainData.getNumStages()
        N_b                                 = sustainData.getNumBiomarkers()
        S_opt                               = S_init.copy()  # have to copy or changes will be passed to S_init
        f_opt                               = np.array(f_init).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        p_perm_k_log                        = np.zeros((M, N + 1, N_S))

        for s in range(N_S):
            shape_S = self._get_shape(S_opt[s])
            p_perm_k_log[:, :, s]               = self._calculate_likelihood_stage(sustainData, S_opt[s], shape_S)

        p_perm_k_weighted_log                   = p_perm_k_log + np.log(f_val_mat)
        # the second summation axis is different to Matlab version
        # adding 1e-250 fixes divide by zero problem that happens rarely
        p_perm_k_norm_log                       = p_perm_k_weighted_log - logsumexp(p_perm_k_weighted_log, axis=(1, 2), keepdims=True)
        p_perm_k_norm                           = np.exp(p_perm_k_norm_log)
        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)
        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))
        order_seq                           = rng.permutation(N_S)
        rep = self.rep_opt

        for s in order_seq:
            order_bio                       = rng.permutation(N_b)
            for i in order_bio:
                current_sequence            = S_opt[s]
                current_shape               = self._get_shape(current_sequence)
                current_sequence_flatten = self._flatten_S_dict(current_sequence)
                assert(current_sequence_flatten.shape[0]==N_b)
                current_location            = np.array([0] * N_b)
                current_location[current_sequence_flatten.astype(int)] = [loc_i for loc_i, size in enumerate(current_shape) for _ in range(size)]
                possible_positions          = np.arange(N)
                possible_sequences          = np.zeros((len(possible_positions), N_b, rep))
                possible_likelihood         = np.zeros((len(possible_positions), rep))
                possible_shapes             = np.zeros((N, self.n_stages, rep))
                possible_p_perm_k_log           = np.zeros((M, N + 1, len(possible_positions), rep))
                for index in range(len(possible_positions)):
                    for r in range(rep):
                        selected_event = i
                        move_event_from = current_location[selected_event]
                        new_sequence = S_opt[s].copy()
                        stage_shape = current_shape.copy()

                        #choose a position in the sequence to move an event to
                        move_event_to           = possible_positions[index]

                        if move_event_from > move_event_to:
                            step = 1
                        elif move_event_from < move_event_to:
                            step = -1
                        else:
                            step = 0
                        if step != 0:
                            if new_sequence[move_event_from].shape[0] > self.min_stage_size:
                                expand_stage = rng.binomial(1, self.p_absorb)
                            else:
                                expand_stage = 0

                            new_sequence[move_event_from] = np.delete(new_sequence[move_event_from],
                                                                    np.where(new_sequence[move_event_from] == selected_event))

                            if not expand_stage:
                                for _ in range(move_event_to, move_event_from, step):
                                    start_cluster = new_sequence[_]
                                    rng.shuffle(start_cluster)
                                    shift_event = start_cluster[0]
                                    new_sequence[_] = np.delete(np.append(start_cluster, selected_event), 0)
                                    selected_event = shift_event
                                new_sequence[_+step] = np.append(new_sequence[_+step], selected_event)
                            else:
                                new_sequence[move_event_to] = np.append(new_sequence[move_event_to], selected_event)
                                stage_shape[move_event_from] -= 1
                                stage_shape[move_event_to] += 1
                        possible_shapes[index,:,r] = stage_shape
                        ns_flatten = self._flatten_S_dict(new_sequence)
                        possible_sequences[index,:,r] = ns_flatten
                        possible_p_perm_k_log[:,:,index,r] = self._calculate_likelihood_stage(sustainData, new_sequence, stage_shape)
                        p_perm_k_log[:,:,s] = possible_p_perm_k_log[:, :, index, r]
                        total_prob_stage_log      = logsumexp(p_perm_k_log + np.log(f_val_mat), axis=2)
                        total_prob_subj_log       = logsumexp(total_prob_stage_log, 1)
                        possible_likelihood[index, r] = np.sum(total_prob_subj_log)

                idx_max, r_max = np.unravel_index(np.argmax(possible_likelihood, axis=None), possible_likelihood.shape)
                this_S = possible_sequences[idx_max, :, r_max].astype(int)
                shape_S = possible_shapes[idx_max, :, r_max]
                S_opt[s] = self._dictionarize_sequence(this_S, shape_S)
                p_perm_k_log[:,:,s] = possible_p_perm_k_log[:,:,idx_max, r_max]
            S_opt[s] = self._dictionarize_sequence(this_S, shape_S)

        p_perm_k_weighted_log               = p_perm_k_log + np.log(f_val_mat)
        p_perm_k_norm_log                   = p_perm_k_weighted_log - logsumexp(p_perm_k_weighted_log, axis=(1, 2), keepdims=True)
        p_perm_k_norm                       = np.exp(p_perm_k_norm_log)
        f_opt                               = (np.squeeze(sum(sum(p_perm_k_norm))) / sum(sum(sum(p_perm_k_norm)))).reshape(N_S, 1, 1)

        f_val_mat                           = np.tile(f_opt, (1, N + 1, M))
        f_val_mat                           = np.transpose(f_val_mat, (2, 1, 0))

        f_opt                               = f_opt.reshape(N_S)
        likelihood_opt                      = np.sum(logsumexp(p_perm_k_log + np.log(f_val_mat), axis=(1,2), keepdims=True))

        return S_opt, f_opt, likelihood_opt

    def _perform_mcmc(self, sustainData, seq_init, f_init, n_iterations, seq_sigma, f_sigma):
        # Take MCMC samples of the uncertainty in the SuStaIn model parameters
        M                                   = sustainData.getNumSamples()
        N                                   = sustainData.getNumStages()
        N_b                                 = sustainData.getNumBiomarkers()
        N_S                                 = len(seq_init)
        shape_S                             = np.vstack([self._get_shape(s) for s in seq_init])
        ss_cumsum = np.cumsum(shape_S, axis=1)
        stage_idx = []
        for _ in ss_cumsum:
            stage_idx_ = {}
            init_idx = 0
            for stage, size in enumerate(_):
                stage_idx_[stage] = np.arange(init_idx, size)
                init_idx = size
            stage_idx.append(stage_idx_)

        seq_init_flatten                    = self._flatten_sequence(seq_init)
        if isinstance(f_sigma, float):
            f_sigma                         = np.array([f_sigma])

        samples_sequence                    = np.zeros((N_S, N_b, n_iterations))
        samples_f                           = np.zeros((N_S, n_iterations))
        samples_likelihood                  = np.zeros((n_iterations, 1))
        samples_sequence[:, :, 0]           = seq_init_flatten  # don't need to copy as we don't write to 0 index
        samples_f[:, 0]                     = f_init
        sample_prob                         = shape_S / shape_S.sum(axis=1).reshape(-1,1)

        # Reduce frequency of tqdm update to 0.1% of total for larger iteration numbers
        tqdm_update_iters = int(n_iterations/1000) if n_iterations > 100000 else None 

        for i in tqdm(range(n_iterations), "MCMC Iteration", n_iterations, miniters=tqdm_update_iters):
            if i > 0:
                # this function returns different random numbers to Matlab

                # Abstract out seq_order loop
                move_event_from_stage = np.array([np.random.choice(np.arange(N).astype(int), 1, p=s)[0] for s in sample_prob])
                move_event_from_idx = np.array([np.random.choice(stage_idx[i][j], 1)[0] for i, j in enumerate(move_event_from_stage)])

                current_sequence = samples_sequence[:, :, i - 1]

                selected_event = current_sequence[np.arange(N_S), move_event_from_idx]

                bm_pos = np.zeros((N_S, N_b))
                for s in range(N_S):
                    bm_pos[s][current_sequence[s].astype(int)] = [loc_i for loc_i, size in enumerate(shape_S[s]) for _ in range(size)]

                distance = bm_pos - move_event_from_stage[:, np.newaxis]

                weight = AbstractSustain.calc_coeff(seq_sigma) * AbstractSustain.calc_exp(distance, 0., seq_sigma)
                weight = np.divide(weight, weight.sum(1)[:, None])

                move_event_to_idx = [self.global_rng.choice(np.arange(N_b), 1, replace=True, p=row)[0] for row in weight]

                # Don't need to copy, but doing it for clarity
                new_seq = current_sequence.copy()
                new_seq[np.arange(N_S), move_event_from_idx] = new_seq[np.arange(N_S), move_event_to_idx]
                new_seq[np.arange(N_S), move_event_to_idx] = selected_event

                samples_sequence[:, :, i] = new_seq

                new_f                       = samples_f[:, i - 1] + f_sigma * self.global_rng.standard_normal()
                new_f                       = (np.fabs(new_f) / np.sum(np.fabs(new_f)))
                samples_f[:, i]             = new_f
            S                               = samples_sequence[:, :, i]

            p_perm_k_log                        = np.zeros((M, N+1, N_S))
            for s in range(N_S):
                S_dict = self._dictionarize_sequence(S[s,:], shape_S[s])
                p_perm_k_log[:,:,s]             = self._calculate_likelihood_stage(sustainData, S_dict, shape_S[s].tolist())


            #NOTE: added extra axes to get np.tile to work the same as Matlab's repmat in this 3D tiling
            f_val_mat                       = np.tile(samples_f[:,i, np.newaxis, np.newaxis], (1, N+1, M))
            f_val_mat                       = np.transpose(f_val_mat, (2, 1, 0))

            total_prob_stage_log            = logsumexp(p_perm_k_log + np.log(f_val_mat), axis=2)
            total_prob_subj_log             = logsumexp(total_prob_stage_log, 1)

            likelihood_sample               = np.sum(total_prob_subj_log)

            samples_likelihood[i]           = likelihood_sample

            if i > 0:
                ratio                           = np.exp(samples_likelihood[i] - samples_likelihood[i - 1])
                if ratio < self.global_rng.random():
                    samples_likelihood[i]       = samples_likelihood[i - 1]
                    samples_sequence[:, :, i]   = samples_sequence[:, :, i - 1]
                    samples_f[:, i]             = samples_f[:, i - 1]

        perm_index                          = np.argmax(samples_likelihood)
        ml_likelihood                       = samples_likelihood[perm_index]
        ml_sequence                         = samples_sequence[:, :, perm_index]
        ml_f                                = samples_f[:, perm_index]

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood

    def _plot_sustain_model(self, *args, **kwargs):
        return sEBMSustain.plot_positional_var(*args, **kwargs)

    def subtype_and_stage_individuals_newData(self, L_yes_new, L_no_new, num_stages, samples_sequence, samples_f, N_samples):

        sustainData_newData               = sEBMSustainData(L_yes_new, L_no_new, num_stages)
        ml_subtype,         \
        prob_ml_subtype,    \
        ml_stage,           \
        prob_ml_stage,      \
        prob_subtype,       \
        prob_stage,         \
        prob_subtype_stage,_          = self.subtype_and_stage_individuals(sustainData_newData, samples_sequence, samples_f, 100)

        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage


    # ********************* STATIC METHODS
    @staticmethod
    def plot_positional_var(ml_sequence_EM, samples_sequence, samples_f, n_samples, biomarker_labels=None, ml_f_EM=None, cval=False, subtype_order=None, biomarker_order=None, title_font_size=12, stage_font_size=10, stage_label="Event Position", stage_rot=0, stage_interval=1, label_font_size=10, label_rot=0, cmap="Oranges", biomarker_colours=None, figsize=None, subtype_titles=None, separate_subtypes=False, save_path=None, save_kwargs={}):
        # Get the number of subtypes
        def _get_shape(S_dict):
            assert type(S_dict) == dict
            N_stages = len(S_dict)
            shape = [len(S_dict[_]) for _ in range(N_stages)]
            return shape
        shape_S = np.vstack([_get_shape(_) for _ in ml_sequence_EM])
        N_S = samples_sequence.shape[0]
        # Get the number of features/biomarkers
        N_bio = samples_sequence.shape[1]
        # Check that the number of labels given match
        if biomarker_labels is not None:
            assert len(biomarker_labels) == N_bio
        # Set subtype order if not given
        if subtype_order is None:
            # Determine order if info given
            if ml_f_EM is not None:
                subtype_order = np.argsort(ml_f_EM)[::-1]
            # Otherwise determine order from samples_f
            else:
                subtype_order = np.argsort(np.mean(samples_f, 1))[::-1]
        # Warn user of reordering if labels and order given
        if biomarker_labels is not None and biomarker_order is not None:
            warnings.warn(
                "Both labels and an order have been given. The labels will be reordered according to the given order!"
            )
        # Use default order if none given
        if biomarker_order is None:
            biomarker_order = np.arange(N_bio)
        # If no labels given, set dummy defaults
        if biomarker_labels is None:
            biomarker_labels = [f"Biomarker {i}" for i in range(N_bio)]
        # Otherwise reorder according to given order (or not if not given)
        else:
            biomarker_labels = [biomarker_labels[i] for i in biomarker_order]
        # Check number of subtype titles is correct if given
        if subtype_titles is not None:
            assert len(subtype_titles) == N_S
        # Check biomarker label colours
        # If custom biomarker text colours are given
        if biomarker_colours is not None:
            biomarker_colours = AbstractSustain.check_biomarker_colours(
            biomarker_colours, biomarker_labels
        )
        # Default case of all-black colours
        # Unnecessary, but skips a check later
        else:
            biomarker_colours = {i:"black" for i in biomarker_labels}

        # Flag to plot subtypes separately
        if separate_subtypes:
            nrows, ncols = 1, 1
        else:
            # Determine number of rows and columns (rounded up)
            if N_S == 1:
                nrows, ncols = 1, 1 # one subtype
            elif N_S < 6:
                nrows, ncols = 1, N_S # one row
            elif N_S < 10:
                nrows, ncols = 2, int(np.ceil(N_S / 2)) # stack into two rows
            else:
                nrows, ncols = 3, int(np.ceil(N_S / 3)) # multiple rows
        # Total axes used to loop over
        total_axes = nrows * ncols
        # Create list of single figure object if not separated
        if separate_subtypes:
            subtype_loops = N_S
        else:
            subtype_loops = 1
        # Container for all figure objects
        figs = []
        # Loop over figures (only makes a diff if separate_subtypes=True)
        text_cmap = plt.cm.get_cmap("Dark2", shape_S.shape[1])
        for i in range(subtype_loops):
            # Create the figure and axis for this subtype loop
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            figs.append(fig)
            # Loop over each axis
            for j in range(total_axes):
                # Normal functionality (all subtypes on one plot)
                if not separate_subtypes:
                    i = j
                # Handle case of a single array
                if isinstance(axs, np.ndarray):
                    ax = axs.flat[i]
                else:
                    ax = axs
                # Turn off axes from rounding up
                if i not in range(N_S):
                    ax.set_axis_off()
                    continue
                this_shape = shape_S[subtype_order[i]]
                shape_cumsum = np.cumsum(this_shape)
                shape_cumsum = np.insert(shape_cumsum,0,0)
                this_samples_sequence = samples_sequence[subtype_order[i],:,:].T
                N = this_samples_sequence.shape[1]

                # Construct confusion matrix (vectorized)
                # We compare `this_samples_sequence` against each position
                # Sum each time it was observed at that point in the sequence
                # And normalize for number of samples/sequences
                confus_matrix = (this_samples_sequence==np.arange(N)[:, None, None]).sum(1) / this_samples_sequence.shape[0]
                confus_matrix_cluster = np.zeros((confus_matrix.shape[0], len(this_shape)))
                for _ in range(shape_cumsum.shape[0] -1):
                    confus_matrix_cluster[:,_] = confus_matrix[:,shape_cumsum[_]:shape_cumsum[_+1]].sum(axis=1)

                if subtype_titles is not None:
                    title_i = subtype_titles[i]
                else:
                    # Add axis title
                    if cval == False:
                        temp_mean_f = np.mean(samples_f, 1)
                        # Shuffle vals according to subtype_order
                        # This defaults to previous method if custom order not given
                        vals = temp_mean_f[subtype_order]

                        if n_samples != np.inf:
                            title_i = f"Subtype {i+1}\n(f={vals[i]:.2f}, n={np.round(vals[i] * n_samples):n})"
                        else:
                            title_i = f"Subtype {i+1}\n(f={vals[i]:.2f})"
                    else:
                        title_i = f"Subtype {i+1}\ncross-validated"

                # Plot the matrix
                # Manually set vmin/vmax to handle edge cases
                # and ensure consistent colourization across figures 
                # when certainty=1
                ax.imshow(
                    confus_matrix_cluster[biomarker_order, :],
                    interpolation='nearest',
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    aspect=0.4
                )
                # Add the xticks and labels
                stage_ticks = np.arange(0, this_shape.shape[0], stage_interval)
                ax.set_xticks(stage_ticks)
                ax.set_xticklabels(stage_ticks+1, fontsize=stage_font_size+10, rotation=stage_rot)
                # Add the yticks and labels
                ax.set_yticks(np.arange(N_bio))
                # Add biomarker labels to LHS of every row
                ax.set_yticklabels(biomarker_labels, ha='right', fontsize=label_font_size+3, rotation=label_rot)
                # Set biomarker label colours
                for t_idx, tick_label in enumerate(ax.get_yticklabels()):
                    clr_idx = np.argmax(t_idx < shape_cumsum)
                    clr = text_cmap(clr_idx-1)
                    tick_label.set_color(clr)
                # Make the event label slightly bigger than the ticks
                ax.set_xlabel(stage_label, fontsize=stage_font_size+8)
                ax.set_title(title_i, fontsize=title_font_size+8)
            # Tighten up the figure
            fig.tight_layout()
            # Save if a path is given
            if save_path is not None:
                # Modify path for specific subtype if specified
                # Don't modify save_path!
                if separate_subtypes:
                    save_name = f"{save_path}_subtype{i+1}"
                else:
                    save_name = f"{save_path}_all-subtypes"
                # Handle file format, avoids issue with . in filenames
                if "format" in save_kwargs:
                    file_format = save_kwargs.pop("format")
                # Default to png
                else:
                    file_format = "png"
                # Save the figure, with additional kwargs
                fig.savefig(
                    f"{save_name}.{file_format}", dpi=300,
                    **save_kwargs
                )
        return figs, axs
        
    @staticmethod
    def linspace_local2(a, b, N, arange_N):
        return a + (b - a) / (N - 1.) * arange_N

    @staticmethod
    def calc_coeff(sig):
        return 1. / np.sqrt(np.pi * 2.0) * sig

    @staticmethod
    def calc_exp(x, mu, sig):
        x = (x - mu) / sig
        return np.exp(-.5 * x * x)

    # ********************* TEST METHODS
    @classmethod
    def test_sustain(cls, n_biomarkers, n_samples, n_subtypes, ground_truth_subtypes, sustain_kwargs, seed=42, mixture_type="mixture_GMM"):
        from pathlib import Path
        # Path to load/save the arrays
        array_path = Path.cwd() / "mixture_arrays.npz"

        # If not present, create and save the mixture arrays
        # NOTE: This will require kde_ebm to be installed, but should not be required by users
        if not Path(array_path).is_file():
            cls.create_mixture_data(n_biomarkers, n_samples, n_subtypes, ground_truth_subtypes, seed, mixture_type, save_path=array_path)
        # Load the saved arrays
        npzfile = np.load("mixture_arrays.npz")
        # Extract the arrays
        L_yes = npzfile['L_yes']
        L_no = npzfile['L_no']
        return cls(
            L_yes, L_no,
            **sustain_kwargs
        )

    def generate_random_model(N_biomarkers, N_S):
        S                                   = np.zeros((N_S, N_biomarkers))
        #try 30 times to find a unique sequence for each subtype
        for i in range(30):
            matched_others                  = False
            for s in range(N_S):
                S[s, :]                     = np.random.permutation(N_biomarkers)
                #compare to all previous sequences
                for i in range(s):
                    if np.all(S[s, :] == S[i, :]):
                        matched_others      = True
            #all subtype sequences are unique, so break
            if not matched_others:
                break
        if matched_others:
            print('WARNING: Iterated 30 times and could not find unique sequences for all subtypes.')
        return S

    @staticmethod
    def generate_data(subtypes, stages, gt_ordering, mixture_style):
        N_biomarkers                        = gt_ordering.shape[1]
        N_subjects                          = len(subtypes)
        #controls are always drawn from N(0, 1) distribution
        mean_controls                       = np.array([0]   * N_biomarkers)
        std_controls                        = np.array([0.25] * N_biomarkers)
        #mean and variance for cases
        #if using mixture_GMM, use normal distribution with mean 1 and std. devs sampled from a range
        if mixture_style == 'mixture_GMM':
            mean_cases                      = np.array([1.5] * N_biomarkers)
            std_cases                       = np.random.uniform(0.25, 0.50, N_biomarkers)
        #if using mixture_KDE, use log normal with mean 0.5 and std devs sampled from a range
        elif mixture_style == 'mixture_KDE':
            mean_cases                      = np.array([0.5] * N_biomarkers)
            std_cases                       = np.random.uniform(0.2, 0.5, N_biomarkers)

        data                                = np.zeros((N_subjects, N_biomarkers))
        data_denoised                       = np.zeros((N_subjects, N_biomarkers))

        stages                              = stages.astype(int)
        #loop over all subjects, creating measurment for each biomarker based on what subtype and stage they're in
        for i in range(N_subjects):
            S_i                             = gt_ordering[subtypes[i], :].astype(int)
            stage_i                         = stages[i].item()

            #fill in with ABNORMAL values up to the subject's stage
            for j in range(stage_i):

                if      mixture_style == 'mixture_KDE':
                    sample_j                = np.random.lognormal(mean_cases[S_i[j]], std_cases[S_i[j]])
                elif    mixture_style == 'mixture_GMM':
                    sample_j                = np.random.normal(mean_cases[S_i[j]], std_cases[S_i[j]])

                data[i, S_i[j]]             = sample_j
                data_denoised[i, S_i[j]]    = mean_cases[S_i[j]]

            # fill in with NORMAL values from the subject's stage+1 to last stage
            for j in range(stage_i, N_biomarkers):
                data[i, S_i[j]]             = np.random.normal(mean_controls[S_i[j]], std_controls[S_i[j]])
                data_denoised[i, S_i[j]]    = mean_controls[S_i[j]]
        return data, data_denoised

    @classmethod
    def create_mixture_data(cls, n_biomarkers, n_samples, n_subtypes, ground_truth_subtypes, seed, mixture_type, save_path):
        # Avoid import outside of testing
        from kde_ebm.mixture_model import fit_all_gmm_models, fit_all_kde_models
        # Set a global seed to propagate (particularly for mixture_model)
        np.random.seed(seed)

        ground_truth_sequences = cls.generate_random_model(n_biomarkers, n_subtypes)

        N_stages = n_biomarkers

        ground_truth_stages_control = np.zeros((int(np.round(n_samples * 0.25)), 1))
        ground_truth_stages_other = np.random.randint(1, N_stages+1, (int(np.round(n_samples * 0.75)), 1))
        ground_truth_stages = np.vstack(
            (ground_truth_stages_control, ground_truth_stages_other)
        ).astype(int)

        data, data_denoised = cls.generate_data(
            ground_truth_subtypes,
            ground_truth_stages,
            ground_truth_sequences,
            mixture_type
        )
        # choose which subjects will be cases and which will be controls
        MIN_CASE_STAGE = np.round((n_biomarkers + 1) * 0.8)
        index_case = np.where(ground_truth_stages >=  MIN_CASE_STAGE)[0]
        index_control = np.where(ground_truth_stages ==  0)[0]

        labels = 2 * np.ones(data.shape[0], dtype=int) # 2 - intermediate value, not used in mixture model fitting
        labels[index_case] = 1                         # 1 - cases
        labels[index_control] = 0                      # 0 - controls

        data_case_control = data[labels != 2, :]
        labels_case_control = labels[labels != 2]
        if mixture_type == "mixture_GMM":
            mixtures = fit_all_gmm_models(data, labels)
        elif mixture_type == "mixture_KDE":
            mixtures = fit_all_kde_models(data, labels)

        L_yes = np.zeros(data.shape)
        L_no = np.zeros(data.shape)
        for i in range(n_biomarkers):
            if mixture_type == "mixture_GMM":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
            elif mixture_type == "mixture_KDE":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1))
        # Save the arrays
        np.savez(save_path, L_yes=L_yes, L_no=L_no)
            
    def cross_validate_sustain_model(self, test_idxs, select_fold = [], plot=False):
        # Cross-validate the SuStaIn model by running the SuStaIn algorithm (E-M
        # and MCMC) on a training dataset and evaluating the model likelihood on a test
        # dataset.
        # Parameters:
        #   'test_idxs'     - list of test set indices for each fold
        #   'select_fold'   - allows user to just run for a single fold (allows the cross-validation to be run in parallel).
        #                     leave this variable empty to iterate across folds sequentially.

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')
        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)

        if select_fold != []:
            if np.isscalar(select_fold):
                select_fold                 = [select_fold]
        else:
            select_fold                     = np.arange(len(test_idxs))
        Nfolds                              = len(select_fold)

        is_full                             = Nfolds == len(test_idxs)

        loglike_matrix                      = np.zeros((Nfolds, self.N_S_max))

        for fold in tqdm(select_fold, "Folds: ", Nfolds, position=0, leave=True):

            indx_test                       = test_idxs[fold]
            indx_train                      = np.array([x for x in range(self.__sustainData.getNumSamples()) if x not in indx_test])

            sustainData_train               = self.__sustainData.reindex(indx_train)
            sustainData_test                = self.__sustainData.reindex(indx_test)

            ml_sequence_prev_EM             = []
            ml_f_prev_EM                    = []

            for s in range(self.N_S_max):

                pickle_filename_fold_s      = os.path.join(pickle_dir, self.dataset_name + '_fold' + str(fold) + '_subtype' + str(s+1) + '.pickle')
                pickle_filepath             = Path(pickle_filename_fold_s)

                if pickle_filepath.exists():

                    print("Loading " + pickle_filename_fold_s)

                    pickle_file             = open(pickle_filename_fold_s, 'rb')

                    loaded_variables        = pickle.load(pickle_file)

                    ml_sequence_EM          = loaded_variables["ml_sequence_EM"]
                    ml_sequence_prev_EM     = loaded_variables["ml_sequence_prev_EM"]
                    ml_f_EM                 = loaded_variables["ml_f_EM"]
                    ml_f_prev_EM            = loaded_variables["ml_f_prev_EM"]

                    samples_likelihood      = loaded_variables["samples_likelihood"]
                    samples_sequence        = loaded_variables["samples_sequence"]
                    samples_f               = loaded_variables["samples_f"]
                    shape_seq               = loaded_variables["shape_seq"]

                    mean_likelihood_subj_test = loaded_variables["mean_likelihood_subj_test"]
                    pickle_file.close()

                    samples_likelihood_subj_test = self._evaluate_likelihood_setofsamples(sustainData_test, shape_seq, samples_sequence, samples_f)

                else:
                    ml_sequence_EM,         \
                    ml_f_EM,                \
                    ml_likelihood_EM,       \
                    ml_sequence_mat_EM,     \
                    ml_f_mat_EM,            \
                    ml_likelihood_mat_EM    = self._estimate_ml_sustain_model_nplus1_clusters(sustainData_train, ml_sequence_prev_EM, ml_f_prev_EM)

                    seq_init                    = ml_sequence_EM
                    f_init                      = ml_f_EM

                    ml_sequence,            \
                    ml_f,                   \
                    ml_likelihood,          \
                    samples_sequence,       \
                    samples_f,              \
                    samples_likelihood           = self._estimate_uncertainty_sustain_model(sustainData_train, seq_init, f_init)
                    
                    shape_seq = np.vstack([self._get_shape(_) for _ in seq_init])

                    samples_likelihood_subj_test = self._evaluate_likelihood_setofsamples(sustainData_test, shape_seq, samples_sequence, samples_f)

                    mean_likelihood_subj_test = np.exp(logsumexp(samples_likelihood_subj_test, axis=1)) / samples_likelihood_subj_test.shape[1]


                    ml_sequence_prev_EM         = ml_sequence_EM
                    ml_f_prev_EM                = ml_f_EM

                    save_variables                                      = {}
                    save_variables["ml_sequence_EM"]                    = ml_sequence_EM
                    save_variables["ml_sequence_prev_EM"]               = ml_sequence_prev_EM
                    save_variables["ml_f_EM"]                           = ml_f_EM
                    save_variables["ml_f_prev_EM"]                      = ml_f_prev_EM

                    save_variables["samples_sequence"]                  = samples_sequence
                    save_variables["samples_f"]                         = samples_f
                    save_variables["samples_likelihood"]                = samples_likelihood

                    save_variables["mean_likelihood_subj_test"]         = mean_likelihood_subj_test
                    save_variables["shape_seq"]                         = shape_seq

                    pickle_file                     = open(pickle_filename_fold_s, 'wb')
                    pickle_output                   = pickle.dump(save_variables, pickle_file)
                    pickle_file.close()

                if is_full:
                    loglike_matrix[fold, s]         = np.mean(np.sum(np.log(samples_likelihood_subj_test),axis=0))

        if not is_full:
            print("Cannot calculate CVIC and loglike_matrix without all folds. Rerun cross_validate_sustain_model after all folds calculated.")
            return [], []

        print(f"Average test set log-likelihood for each subtype model: {np.mean(loglike_matrix, 0)}")

        if plot:
            import pandas as pd
            fig, ax = plt.subplots()

            df_loglike = pd.DataFrame(data = loglike_matrix, columns = ["Subtype " + str(i+1) for i in range(self.N_S_max)])
            df_loglike.boxplot(grid=False, ax=ax, fontsize=15)
            for i in range(self.N_S_max):
                y = df_loglike[["Subtype " + str(i+1)]]
                # Add some random "jitter" to the x-axis
                x = np.random.normal(1+i, 0.04, size=len(y))
                ax.plot(x, y.values, 'r.', alpha=0.2)
            fig.savefig(Path(self.output_folder) / 'Log_likelihoods_cv_folds.png')
            fig.show()

        CVIC                            = np.zeros(self.N_S_max)

        for s in range(self.N_S_max):
            for fold in range(Nfolds):
                pickle_filename_fold_s  = os.path.join(pickle_dir, self.dataset_name + '_fold' + str(fold) + '_subtype' + str(s+1) + '.pickle')
                pickle_filepath         = Path(pickle_filename_fold_s)

                pickle_file             = open(pickle_filename_fold_s, 'rb')
                loaded_variables        = pickle.load(pickle_file)

                mean_likelihood_subj_test = loaded_variables["mean_likelihood_subj_test"]
                pickle_file.close()
    
                if fold == 0:
                    mean_likelihood_subj_test_cval    = mean_likelihood_subj_test
                else:
                    mean_likelihood_subj_test_cval    = np.concatenate((mean_likelihood_subj_test_cval, mean_likelihood_subj_test), axis=0)

            CVIC[s]                     = -2*sum(np.log(mean_likelihood_subj_test_cval))

        print("CVIC for each subtype model: " + str(CVIC))

        return CVIC, loglike_matrix


    def combine_cross_validated_sequences(self, N_subtypes, N_folds, plot_format="png", **kwargs):
        # Combine MCMC sequences across cross-validation folds to get cross-validated positional variance diagrams,
        # so that you get more realistic estimates of variance within event positions within subtypes

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')

        #*********** load ML sequence for full model for N_subtypes
        pickle_filename_s                   = os.path.join(pickle_dir, self.dataset_name + '_subtype' + str(N_subtypes) + '.pickle')        
        pickle_filepath                     = Path(pickle_filename_s)

        assert pickle_filepath.exists(), "Failed to find pickle file for full model with " + str(N_subtypes) + " subtypes."

        pickle_file                         = open(pickle_filename_s, 'rb')

        loaded_variables_full               = pickle.load(pickle_file)

        ml_sequence_EM_full                 = loaded_variables_full["ml_sequence_EM"]
        ml_f_EM_full                        = loaded_variables_full["ml_f_EM"]

        for i in range(N_folds):

            #load the MCMC sequences for this fold's model of N_subtypes
            pickle_filename_fold_s          = os.path.join(pickle_dir, self.dataset_name + '_fold' + str(i) + '_subtype' + str(N_subtypes) + '.pickle')        
            pickle_filepath                 = Path(pickle_filename_fold_s)

            assert pickle_filepath.exists(), "Failed to find pickle file for fold " + str(i)

            pickle_file                     = open(pickle_filename_fold_s, 'rb')

            loaded_variables_i              = pickle.load(pickle_file)

            ml_sequence_EM_i                = loaded_variables_i["ml_sequence_EM"]
            ml_f_EM_i                       = loaded_variables_i["ml_f_EM"]

            samples_sequence_i              = loaded_variables_i["samples_sequence"]
            samples_f_i                     = loaded_variables_i["samples_f"]

            mean_likelihood_subj_test       = loaded_variables_i["mean_likelihood_subj_test"]

            pickle_file.close()

            # Really simple approach: choose order based on this fold's fraction of subjects per subtype
            # It doesn't work very well when the fractions of subjects are similar across subtypes
            #mean_f_i                        = np.mean(samples_f_i, 1)
            #iMax_vec                        = np.argsort(mean_f_i)[::-1]
            #iMax_vec                        = iMax_vec.astype(int)

            #This approach seems to work better:
            # 1. calculate the Kendall's tau correlation matrix,
            # 2. Flatten the matrix into a vector
            # 3. Sort the vector, then unravel the flattened indices back into matrix style (x, y) indices
            # 4. Find the order in which this fold's subtypes first appear in the sorted list
            corr_mat                        = np.zeros((N_subtypes, N_subtypes))
            for j in range(N_subtypes):
                for k in range(N_subtypes):
                    corr_mat[j,k]            = stats.kendalltau(ml_sequence_EM_full[j,:], ml_sequence_EM_i[k,:]).correlation
            set_full                        = []
            set_fold_i                      = []
            i_i, i_j                        = np.unravel_index(np.argsort(corr_mat.flatten())[::-1], (N_subtypes, N_subtypes))
            for k in range(len(i_i)):
                if not i_i[k] in set_full and not i_j[k] in set_fold_i:
                    set_full.append(i_i[k].astype(int))
                    set_fold_i.append(i_j[k].astype(int))
            index_set_full                  = np.argsort(set_full).astype(int)
            iMax_vec                        = [set_fold_i[i] for i in index_set_full]

            assert(np.all(np.sort(iMax_vec)==np.arange(N_subtypes)))

            if i == 0:
                samples_sequence_cval       = samples_sequence_i[iMax_vec,:,:]
                samples_f_cval              = samples_f_i[iMax_vec, :]
            else:
                samples_sequence_cval       = np.concatenate((samples_sequence_cval,    samples_sequence_i[iMax_vec,:,:]),  axis=2)
                samples_f_cval              = np.concatenate((samples_f_cval,           samples_f_i[iMax_vec,:]),           axis=1)

        n_samples                           = self.__sustainData.getNumSamples()

        plot_subtype_order                  = np.argsort(ml_f_EM_full)[::-1]
        # order of biomarkers in each subtypes' positional variance diagram
        plot_biomarker_order                = ml_sequence_EM_full[plot_subtype_order[0], :].astype(int)

        figs, ax = self._plot_sustain_model(
            samples_sequence=samples_sequence_cval,
            samples_f=samples_f_cval,
            n_samples=n_samples,
            cval=True,
            biomarker_labels=self.biomarker_labels,
            subtype_order=plot_subtype_order,
            biomarker_order=plot_biomarker_order,
            **kwargs
        )
        # If saving is being done here
        if "save_path" not in kwargs:
            # Handle separated subtypes
            if len(figs) > 1:
                # Loop over each figure/subtype
                for num_subtype, fig in zip(range(N_subtypes), figs):
                    # Nice confusing filename
                    plot_fname = Path(
                        self.output_folder
                    ) / f"{self.dataset_name}_subtype{N_subtypes - 1}_subtype{num_subtype}-separated_PVD_{N_folds}fold_CV.{plot_format}"
                    # Save the figure
                    fig.savefig(plot_fname, bbox_inches='tight')
                    fig.show()
            # Otherwise default single plot
            else:
                fig = figs[0]
                # save and show this figure after all subtypes have been calculcated
                plot_fname = Path(
                    self.output_folder
                ) / f"{self.dataset_name}_subtype{N_subtypes - 1}_PVD_{N_folds}fold_CV.{plot_format}"
                # Save the figure
                fig.savefig(plot_fname, bbox_inches='tight')
                fig.show()

        #return samples_sequence_cval, samples_f_cval, kendalls_tau_mat, f_mat #samples_sequence_cval

    def subtype_and_stage_individuals(self, sustainData, shape_seq, samples_sequence, samples_f, N_samples):
        # Subtype and stage a set of subjects. Useful for subtyping/staging subjects that were not used to build the model

        nSamples                            = sustainData.getNumSamples()
        nStages                             = sustainData.getNumStages()

        n_iterations_MCMC                   = samples_sequence.shape[2]
        select_samples                      = np.round(np.linspace(0, n_iterations_MCMC - 1, N_samples))
        N_S                                 = samples_sequence.shape[0]
        temp_mean_f                         = np.mean(samples_f, axis=1)
        ix                                  = np.argsort(temp_mean_f)[::-1]
        assert shape_seq.shape == (N_S, nStages), "Shape array should correspond to number of subtypes (rows) and number of stages (cols)"
        shape_seq                           = shape_seq[ix,:]

        prob_subtype_stage                  = np.zeros((nSamples, nStages + 1, N_S))
        prob_subtype                        = np.zeros((nSamples, N_S))
        prob_stage                          = np.zeros((nSamples, nStages + 1))
        ll_list = []

        for i in range(N_samples):
            sample                          = int(select_samples[i])

            this_S                          = samples_sequence[ix, :, sample]
            this_S_dict                     = [self._dictionarize_sequence(this_S[i_], S_) for i_, S_ in enumerate(shape_seq)]
            this_f                          = samples_f[ix, sample]

            ll,                  \
            _,                  \
            total_prob_stage_log,   \
            total_prob_subtype_log, \
            total_prob_subtype_stage_log        = self._calculate_likelihood(sustainData, this_S_dict, this_f)

            total_prob_subtype_log              = total_prob_subtype_log.reshape(len(total_prob_subtype_log), N_S)
            total_prob_subtype_norm_log         = total_prob_subtype_log     - np.tile(logsumexp(total_prob_subtype_log, axis=1).reshape(len(total_prob_subtype_log), 1),        (1, N_S))
            total_prob_subtype_norm         = np.exp(total_prob_subtype_norm_log)
            total_prob_stage_norm_log           = total_prob_stage_log   - np.tile(logsumexp(total_prob_stage_log, axis=1).reshape(len(total_prob_stage_log), 1),          (1, nStages + 1)) #removed total_prob_subtype
            total_prob_stage_norm       = np.exp(total_prob_stage_norm_log)
            total_prob_subtype_stage_norm_log   = total_prob_subtype_stage_log - np.tile(logsumexp(total_prob_subtype_stage_log, axis=(1,2),keepdims=True).reshape(nSamples, 1, 1),(1, nStages + 1, N_S))
            total_prob_subtype_stage_norm        = np.exp(total_prob_subtype_stage_norm_log)

            prob_subtype_stage              = (i / (i + 1.) * prob_subtype_stage)   + (1. / (i + 1.) * total_prob_subtype_stage_norm)
            prob_subtype                    = (i / (i + 1.) * prob_subtype)         + (1. / (i + 1.) * total_prob_subtype_norm)
            prob_stage                      = (i / (i + 1.) * prob_stage)           + (1. / (i + 1.) * total_prob_stage_norm)
            ll_list.append(ll)

        ml_subtype                          = np.nan * np.ones((nSamples, 1))
        prob_ml_subtype                     = np.nan * np.ones((nSamples, 1))
        ml_stage                            = np.nan * np.ones((nSamples, 1))
        prob_ml_stage                       = np.nan * np.ones((nSamples, 1))

        for i in range(nSamples):
            this_prob_subtype               = np.squeeze(prob_subtype[i, :])
            if (np.sum(np.isnan(this_prob_subtype)) == 0):
                this_subtype                = np.where(np.atleast_1d(this_prob_subtype) == np.max(this_prob_subtype))
                try:
                    ml_subtype[i]           = this_subtype
                except:
                    ml_subtype[i]           = this_subtype[0][0]
                if this_prob_subtype.size == 1 and this_prob_subtype == 1:
                    prob_ml_subtype[i]      = 1
                else:
                    try:
                        prob_ml_subtype[i]  = this_prob_subtype[this_subtype]
                    except:
                        prob_ml_subtype[i]  = this_prob_subtype[this_subtype[0][0]]
            this_prob_stage                 = np.squeeze(prob_subtype_stage[i, :, int(ml_subtype[i])])

            if (np.sum(np.isnan(this_prob_stage)) == 0):
                this_stage                  = np.where(this_prob_stage == np.max(this_prob_stage))
                ml_stage[i]                 = this_stage[0][0]
                prob_ml_stage[i]            = this_prob_stage[this_stage[0][0]]
        # NOTE: The above loop can be replaced with some simpler numpy calls
        # May need to do some masking to avoid NaNs, or use `np.nanargmax` depending on preference
        # E.g. ml_subtype == prob_subtype.argmax(1)
        # E.g. ml_stage == prob_subtype_stage[np.arange(prob_subtype_stage.shape[0]), :, ml_subtype].argmax(1)
        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage, ll_list

    def _estimate_ml_sustain_model_nplus1_clusters(self, sustainData, ml_sequence_prev, ml_f_prev):
        # Given the previous SuStaIn model, estimate the next model in the
        # hierarchy (i.e. number of subtypes goes from N to N+1)
        #
        #
        # OUTPUTS:
        # ml_sequence       - the ordering of the stages for each subtype for the next SuStaIn model in the hierarchy
        # ml_f              - the most probable proportion of individuals belonging to each subtype for the next SuStaIn model in the hierarchy
        # ml_likelihood     - the likelihood of the most probable SuStaIn model for the next SuStaIn model in the hierarchy

        N_S = len(ml_sequence_prev) + 1
        if N_S == 1:
            # If the number of subtypes is 1, fit a single linear z-score model
            print('Finding ML solution to 1 cluster problem')
            start_time = time.time()
            ml_sequence,        \
            ml_f,               \
            ml_likelihood,      \
            ml_sequence_mat,    \
            ml_f_mat,           \
            ml_likelihood_mat   = self._find_ml(sustainData)
            end_time = time.time()
            print('Overall ML likelihood is', ml_likelihood)

        else:
            # If the number of subtypes is greater than 1, go through each subtype
            # in turn and try splitting into two subtypes
            start_time = time.time()
            _, _, _, p_sequence_log, _          = self._calculate_likelihood(sustainData, ml_sequence_prev, ml_f_prev)

            p_sequence_log                  = p_sequence_log.reshape(p_sequence_log.shape[0], N_S - 1)
            p_sequence_norm_log                 = p_sequence_log - np.tile(logsumexp(p_sequence_log, axis=1).reshape(len(p_sequence_log), 1), (N_S - 1))
            p_sequence_norm             = np.exp(p_sequence_norm_log)

            ml_cluster_subj                 = np.zeros((sustainData.getNumSamples(), 1))
            for m in range(sustainData.getNumSamples()):
                ix                          = np.argmax(p_sequence_norm[m, :]) + 1
                ml_cluster_subj[m]          = ix

            ml_likelihood                   = -np.inf
            for ix_cluster_split in range(N_S - 1):
                this_N_cluster              = sum(ml_cluster_subj == int(ix_cluster_split + 1))

                if this_N_cluster > 1:

                    # Take the data from the individuals belonging to a particular
                    # cluster and fit a two subtype model
                    print('Splitting cluster', ix_cluster_split + 1, 'of', N_S - 1)
                    ix_i                    = (ml_cluster_subj == int(ix_cluster_split + 1)).reshape(sustainData.getNumSamples(), )
                    sustainData_i           = sustainData.reindex(ix_i)

                    print(' + Resolving 2 cluster problem')
                    this_ml_sequence_split, _, _, _, _, _ = self._find_ml_split(sustainData_i)

                    # Use the two subtype model combined with the other subtypes to
                    # inititialise the fitting of the next SuStaIn model in the
                    # hierarchy
                    this_seq_init           = ml_sequence_prev.copy()  # have to copy or changes will be passed to ml_sequence_prev

                    #replace the previous sequence with the first (row index zero) new sequence
                    this_seq_init[ix_cluster_split] = this_ml_sequence_split[0]

                    #add the second new sequence (row index one) to the stack of sequences, 
                    #so that you now have N_S sequences instead of N_S-1
                    this_seq_init.append(this_ml_sequence_split[1])

                    #initialize fraction of subjects in each subtype to be uniform
                    this_f_init             = np.array([1.] * N_S) / float(N_S)

                    print(' + Finding ML solution from hierarchical initialisation')
                    this_ml_sequence,       \
                    this_ml_f,              \
                    this_ml_likelihood,     \
                    this_ml_sequence_mat,   \
                    this_ml_f_mat,          \
                    this_ml_likelihood_mat  = self._find_ml_mixture(sustainData, this_seq_init, this_f_init)

                    # Choose the most probable SuStaIn model from the different
                    # possible SuStaIn models initialised by splitting each subtype
                    # in turn
                    if this_ml_likelihood[0] > ml_likelihood:
                        ml_likelihood       = this_ml_likelihood[0]
                        ml_sequence         = this_ml_sequence
                        ml_f                = this_ml_f
                        ml_likelihood_mat   = this_ml_likelihood_mat
                        ml_sequence_mat     = this_ml_sequence_mat
                        ml_f_mat            = this_ml_f_mat
                    print('- ML likelihood is', this_ml_likelihood[0])
                else:
                    print(f'Cluster {ix_cluster_split + 1} of {N_S - 1} too small for subdivision')
            print(f'Overall ML likelihood is', ml_likelihood)
            end_time = time.time()
        run_time = end_time - start_time

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat, run_time

    def _find_ml(self, sustainData):
        # Fit the maximum likelihood model
        #
        # OUTPUTS:
        # ml_sequence   - the ordering of the stages for each subtype
        # ml_f          - the most probable proportion of individuals belonging to each subtype
        # ml_likelihood - the likelihood of the most probable SuStaIn model
        partial_iter = partial(self._find_ml_iteration, sustainData)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list                    = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if ~isinstance(pool_output_list, list):
            pool_output_list                = list(pool_output_list)

        ml_sequence_list                    = []
        ml_f_mat                            = np.zeros((1, self.N_startpoints))
        ml_likelihood_mat                   = np.zeros(self.N_startpoints)

        for i in range(self.N_startpoints):
            ml_sequence_list.append(pool_output_list[i][0])
            ml_f_mat[:, i]                  = pool_output_list[i][1]
            ml_likelihood_mat[i]            = pool_output_list[i][2]

        ix                                  = np.argmax(ml_likelihood_mat)
        ml_sequence                         = ml_sequence_list[ix]
        ml_f                                = ml_f_mat[:, ix]
        ml_likelihood                       = ml_likelihood_mat[ix]
        return ml_sequence, ml_f, ml_likelihood, ml_sequence_list, ml_f_mat, ml_likelihood_mat

    def _find_ml_iteration(self, sustainData, seed_seq):
        rng = np.random.default_rng(seed_seq)

        # randomly initialise the sequence of the linear z-score model
        seq_init                        = self._initialise_sequence(sustainData, rng)
        f_init                          = [1]

        this_ml_sequence,   \
        this_ml_f,          \
        this_ml_likelihood, \
        _,                  \
        _,                  \
        _,                  \
        _   = self._perform_em(sustainData, seq_init, f_init, rng)

        return this_ml_sequence, this_ml_f, this_ml_likelihood

    #********************************************

    def _find_ml_split(self, sustainData):
        # Fit a mixture of two models
        #
        #
        # OUTPUTS:
        # ml_sequence   - the ordering of the stages for each subtype
        # ml_f          - the most probable proportion of individuals belonging to each subtype
        # ml_likelihood - the likelihood of the most probable SuStaIn model

        N_S                                 = 2

        partial_iter                        = partial(self._find_ml_split_iteration, sustainData)
        seed_sequences = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list                    = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if ~isinstance(pool_output_list, list):
            pool_output_list                = list(pool_output_list)

        ml_sequence_mat                     = []
        ml_f_mat                            = np.zeros((N_S, self.N_startpoints))
        ml_likelihood_mat                   = np.zeros((self.N_startpoints, 1))

        for i in range(self.N_startpoints):
            ml_sequence_mat.append(pool_output_list[i][0])
            ml_f_mat[:, i]                  = pool_output_list[i][1]
            ml_likelihood_mat[i]            = pool_output_list[i][2]

        ix                                  = np.argmax(ml_likelihood_mat)

        ml_sequence                         = ml_sequence_mat[ix]
        ml_f                                = ml_f_mat[:, ix]
        ml_likelihood                       = ml_likelihood_mat[ix]

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    def _find_ml_split_iteration(self, sustainData, seed_seq):
        # Get process-appropriate Generator
        rng = np.random.default_rng(seed_seq)

        N_S                                 = 2

        # randomly initialise individuals as belonging to one of the two subtypes (clusters)
        min_N_cluster                       = 0
        while min_N_cluster == 0:
            vals = rng.random(sustainData.getNumSamples())
            cluster_assignment = np.ceil(N_S * vals).astype(int)
            # Count cluster sizes
            # Guarantee 1s and 2s counts with minlength=3
            # Ignore 0s count with [1:]
            cluster_sizes = np.bincount(cluster_assignment, minlength=3)[1:]
            # Get the minimum cluster size
            min_N_cluster = cluster_sizes.min()

        # initialise the stages of the two models by fitting a single model to each of the two sets of individuals
        seq_init                            = []
        for s in range(N_S):
            index_s                         = cluster_assignment.reshape(cluster_assignment.shape[0], ) == (s + 1)
            temp_sustainData                = sustainData.reindex(index_s)

            temp_seq_init                   = self._initialise_sequence(sustainData, rng)
            seq_init_split, _, _, _, _, _, _  = self._perform_em(temp_sustainData, temp_seq_init, [1], rng)
            seq_init.append(seq_init_split[0])

        f_init                              = np.array([1.] * N_S) / float(N_S)

        # optimise the mixture of two models from the initialisation
        this_ml_sequence, \
        this_ml_f, \
        this_ml_likelihood, _, _, _, _    = self._perform_em(sustainData, seq_init, f_init, rng)

        return this_ml_sequence, this_ml_f, this_ml_likelihood

    #********************************************
    def _find_ml_mixture(self, sustainData, seq_init, f_init):
        # Fit a mixture of models
        #
        #
        # OUTPUTS:
        # ml_sequence   - the ordering of the stages for each subtype for the next SuStaIn model in the hierarchy
        # ml_f          - the most probable proportion of individuals belonging to each subtype for the next SuStaIn model in the hierarchy
        # ml_likelihood - the likelihood of the most probable SuStaIn model for the next SuStaIn model in the hierarchy

        N_S                                 = len(seq_init)
        partial_iter                        = partial(self._find_ml_mixture_iteration, sustainData, seq_init, f_init)
        seed_sequences                      = np.random.SeedSequence(self.global_rng.integers(1e10))
        pool_output_list                    = self.pool.map(partial_iter, seed_sequences.spawn(self.N_startpoints))

        if ~isinstance(pool_output_list, list):
            pool_output_list                = list(pool_output_list)

        ml_sequence_mat                     = []
        ml_f_mat                            = np.zeros((N_S, self.N_startpoints))
        ml_likelihood_mat                   = np.zeros((self.N_startpoints, 1))

        for i in range(self.N_startpoints):
            ml_sequence_mat.append(pool_output_list[i][0])
            ml_f_mat[:, i]                  = pool_output_list[i][1]
            ml_likelihood_mat[i]            = pool_output_list[i][2]

        ix                                  = np.argmax(ml_likelihood_mat)

        ml_sequence                         = ml_sequence_mat[ix]
        ml_f                                = ml_f_mat[:, ix]
        ml_likelihood                       = ml_likelihood_mat[ix]

        return ml_sequence, ml_f, ml_likelihood, ml_sequence_mat, ml_f_mat, ml_likelihood_mat

    def _find_ml_mixture_iteration(self, sustainData, seq_init, f_init, seed_seq):
        # Get process-appropriate Generator
        rng = np.random.default_rng(seed_seq)

        ml_sequence,        \
        ml_f,               \
        ml_likelihood,      \
        samples_sequence,   \
        samples_f,          \
        samples_likelihood, \
        samples_shapes           = self._perform_em(sustainData, seq_init, f_init, rng)

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood, samples_shapes
    #********************************************

    def _perform_em(self, sustainData, current_sequence, current_f, rng):

        # Perform an E-M procedure to estimate parameters of SuStaIn model
        MaxIter                             = self.N_em

        N                                   = sustainData.getNumStages()
        N_b                                 = sustainData.getNumBiomarkers()
        N_S                                 = len(current_sequence)
        current_likelihood, _, _, _, _      = self._calculate_likelihood(sustainData, current_sequence, current_f)

        terminate                           = 0
        iteration                           = 0
        samples_sequence                    = np.nan * np.ones((MaxIter, N_b, N_S))
        samples_f                           = np.nan * np.ones((MaxIter, N_S))
        samples_likelihood                  = np.nan * np.ones((MaxIter, 1))
        samples_stage_size                  = np.nan * np.ones((MaxIter, self.n_stages, N_S))
        shapes_init                         = np.vstack([self._get_shape(current_s) for current_s in current_sequence])
        assert shapes_init.shape[1] == self.n_stages
        current_sequence_array = np.vstack([self._flatten_S_dict(S_dict) for S_dict in current_sequence])
        samples_sequence[0, :, :] = current_sequence_array.T
        current_f                           = np.array(current_f).reshape(len(current_f))
        samples_f[0, :]                     = current_f
        samples_likelihood[0]               = current_likelihood
        samples_stage_size[0,:,:]           = shapes_init.T
        while terminate == 0:

            candidate_sequence,     \
            candidate_f,            \
            candidate_likelihood            = self._optimise_parameters(sustainData, current_sequence, current_f, rng)

            HAS_converged                   = np.fabs((candidate_likelihood - current_likelihood) / max(candidate_likelihood, current_likelihood)) < 1e-6
            if HAS_converged:
                terminate                   = 1
            else:
                if candidate_likelihood > current_likelihood:
                    current_sequence        = candidate_sequence
                    current_f               = candidate_f
                    current_likelihood      = candidate_likelihood

            current_sequence_array = np.vstack([self._flatten_S_dict(S_dict) for S_dict in current_sequence])
            current_sequence_shape = np.vstack([self._get_shape(S_dict) for S_dict in current_sequence])
            samples_sequence[iteration, :, :] = current_sequence_array.T
            samples_f[iteration, :]         = current_f
            samples_likelihood[iteration]   = current_likelihood
            samples_stage_size[iteration,:,:] = current_sequence_shape.T

            if iteration == (MaxIter - 1):
                terminate                   = 1
            iteration                       = iteration + 1

        ml_sequence                         = current_sequence
        ml_f                                = current_f
        ml_likelihood                       = current_likelihood
        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood, samples_stage_size



    def _estimate_uncertainty_sustain_model(self, sustainData, seq_init, f_init):
        # Estimate the uncertainty in the subtype progression patterns and
        # proportion of individuals belonging to the SuStaIn model
        #
        #
        # OUTPUTS:
        # ml_sequence       - the most probable ordering of the stages for each subtype found across MCMC samples
        # ml_f              - the most probable proportion of individuals belonging to each subtype found across MCMC samples
        # ml_likelihood     - the likelihood of the most probable SuStaIn model found across MCMC samples
        # samples_sequence  - samples of the ordering of the stages for each subtype obtained from MCMC sampling
        # samples_f         - samples of the proportion of individuals belonging to each subtype obtained from MCMC sampling
        # samples_likeilhood - samples of the likelihood of each SuStaIn model sampled by the MCMC sampling

        # Perform a few initial passes where the perturbation sizes of the MCMC uncertainty estimation are tuned
        start_time = time.time()
        seq_sigma_opt, f_sigma_opt          = self._optimise_mcmc_settings(sustainData, seq_init, f_init)

        # Run the full MCMC algorithm to estimate the uncertainty
        
        ml_sequence,        \
        ml_f,               \
        ml_likelihood,      \
        samples_sequence,   \
        samples_f,          \
        samples_likelihood                  = self._perform_mcmc(sustainData, seq_init, f_init, self.N_iterations_MCMC, seq_sigma_opt, f_sigma_opt)
        end_time = time.time()
        mcmc_time = end_time - start_time

        return ml_sequence, ml_f, ml_likelihood, samples_sequence, samples_f, samples_likelihood, mcmc_time

    def _optimise_mcmc_settings(self, sustainData, seq_init, f_init):

        # Optimise the perturbation size for the MCMC algorithm
        n_iterations_MCMC_optimisation      = int(self.N_iterations_MCMC_init)
        n_passes_optimisation               = 3
        seq_sigma_currentpass               = 1
        f_sigma_currentpass                 = 0.01
        N_S                                 = len(seq_init)
        shape_S = np.vstack([self._get_shape(s) for s in seq_init])
        assert shape_S.shape == (N_S, self.n_stages)
        assert all(shape_S.sum(axis=1) == sustainData.getNumBiomarkers())
        for i in range(n_passes_optimisation):
            _, _, _, samples_sequence_currentpass, samples_f_currentpass, _ = self._perform_mcmc(   sustainData,
                                                                                                     seq_init,
                                                                                                     f_init,
                                                                                                     n_iterations_MCMC_optimisation,
                                                                                                     seq_sigma_currentpass,
                                                                                                     f_sigma_currentpass)

            samples_position_currentpass    = np.zeros(samples_sequence_currentpass.shape)
            for s in range(N_S):
                for sample in range(n_iterations_MCMC_optimisation):
                    temp_seq                        = samples_sequence_currentpass[s, :, sample]
                    temp_inv                        = np.array([0] * samples_sequence_currentpass.shape[1])
                    temp_inv[temp_seq.astype(int)]  = [loc_i for loc_i, size in enumerate(shape_S[s]) for _ in range(size)]
                    samples_position_currentpass[s, :, sample] = temp_inv

            seq_sigma_currentpass           = np.std(samples_position_currentpass, axis=2, ddof=1)
            seq_sigma_currentpass[seq_sigma_currentpass < 1.0] = 1.0  # magic number
            f_sigma_currentpass             = np.std(samples_f_currentpass, axis=1, ddof=1)

        seq_sigma_opt                       = seq_sigma_currentpass
        f_sigma_opt                         = f_sigma_currentpass

        return seq_sigma_opt, f_sigma_opt

    def _evaluate_likelihood_setofsamples(self, sustainData, shape_seq, samples_sequence, samples_f):
        n_total                             = samples_sequence.shape[2]
        #reduce the number of samples to speed this function up
        if n_total >= 1e6:
            N_samples                       = int(np.round(n_total/1000))
        elif n_total >= 1e5:
            N_samples                       = int(np.round(n_total/100))
        else:
            N_samples                       = n_total
        select_samples                      = np.round(np.linspace(0, n_total - 1, N_samples)).astype(int)
        samples_sequence                    = samples_sequence[:, :, select_samples]
        samples_f                           = samples_f[:, select_samples]
        # Take MCMC samples of the uncertainty in the SuStaIn model parameters
        M                                   = sustainData.getNumSamples()
        n_iterations                        = samples_sequence.shape[2]
        samples_likelihood_subj             = np.zeros((M, n_iterations))
        for i in range(n_iterations):
            S                               = samples_sequence[:, :, i]
            f                               = samples_f[:, i]
            this_S_dict                     = [self._dictionarize_sequence(S[i_], shape) for i_, shape in enumerate(shape_seq)]

            _, likelihood_sample_subj, _, _, _  = self._calculate_likelihood(sustainData, this_S_dict, f)

            samples_likelihood_subj[:, i]   = likelihood_sample_subj

        return samples_likelihood_subj


    # Externally called method to start the SuStaIn algorithm after initializing the SuStaIn class object properly
    def run_sustain_algorithm(self, plot=False, plot_format="png", **kwargs):
        ml_sequence_prev_EM                 = []
        ml_f_prev_EM                        = []

        pickle_dir                          = os.path.join(self.output_folder, 'pickle_files')
        if not os.path.isdir(pickle_dir):
            os.mkdir(pickle_dir)
        if plot:
            fig0, ax0                           = plt.subplots()
        for s in range(self.N_S_max):

            pickle_filename_s               = os.path.join(pickle_dir, self.dataset_name + '_subtype' + str(s+1) + '.pickle')
            pickle_filepath                 = Path(pickle_filename_s)
            if pickle_filepath.exists():
                print("Found pickle file: " + pickle_filename_s + ". Using pickled variables for " + str(s+1) + " subtype" + ("s" if s > 0 else "") + ".")

                pickle_file                 = open(pickle_filename_s, 'rb')

                loaded_variables            = pickle.load(pickle_file)

                samples_likelihood          = loaded_variables["samples_likelihood"]
                samples_sequence            = loaded_variables["samples_sequence"]
                samples_f                   = loaded_variables["samples_f"]

                ml_sequence_EM              = loaded_variables["ml_sequence_EM"]
                ml_sequence_prev_EM         = loaded_variables["ml_sequence_prev_EM"]
                ml_f_EM                     = loaded_variables["ml_f_EM"]
                ml_f_prev_EM                = loaded_variables["ml_f_prev_EM"]

                pickle_file.close()
            else:
                print("Failed to find pickle file: " + pickle_filename_s + ". Running SuStaIn model for " + str(s+1) + " subtype" + ("s" if s > 0 else "") + ".")

                ml_sequence_EM,     \
                ml_f_EM,            \
                ml_likelihood_EM,   \
                ml_sequence_mat_EM, \
                ml_f_mat_EM,        \
                ml_likelihood_mat_EM, opt_time  = self._estimate_ml_sustain_model_nplus1_clusters(self.__sustainData, ml_sequence_prev_EM, 
                                                                                              ml_f_prev_EM)

                seq_init                    = ml_sequence_EM
                f_init                      = ml_f_EM

                ml_sequence,        \
                ml_f,               \
                ml_likelihood,      \
                samples_sequence,   \
                samples_f,          \
                samples_likelihood, mcmc_time          = self._estimate_uncertainty_sustain_model(self.__sustainData, seq_init, f_init)
                ml_sequence_prev_EM         = ml_sequence_EM
                ml_f_prev_EM                = ml_f_EM

            # max like subtype and stage / subject
            shape_S = np.vstack([self._get_shape(_) for _ in seq_init])
            N_samples                       = 1000
            ml_subtype,             \
            prob_ml_subtype,        \
            ml_stage,               \
            prob_ml_stage,          \
            prob_subtype,           \
            prob_stage,             \
            prob_subtype_stage, ll               = self.subtype_and_stage_individuals(self.__sustainData, shape_S, samples_sequence, samples_f, N_samples)   #self.subtype_and_stage_individuals(self.__data, samples_sequence, samples_f, N_samples)
            if not pickle_filepath.exists():

                if not os.path.exists(self.output_folder):
                    os.makedirs(self.output_folder)

                save_variables                          = {}
                save_variables["samples_sequence"]      = samples_sequence
                save_variables["samples_f"]             = samples_f
                save_variables["samples_likelihood"]    = samples_likelihood

                save_variables["ml_subtype"]            = ml_subtype
                save_variables["prob_ml_subtype"]       = prob_ml_subtype
                save_variables["ml_stage"]              = ml_stage
                save_variables["prob_ml_stage"]         = prob_ml_stage
                save_variables["prob_subtype"]          = prob_subtype
                save_variables["prob_stage"]            = prob_stage
                save_variables["prob_subtype_stage"]    = prob_subtype_stage

                save_variables["ml_sequence_EM"]        = ml_sequence_EM
                save_variables["ml_sequence_prev_EM"]   = ml_sequence_prev_EM
                save_variables["ml_f_EM"]               = ml_f_EM
                save_variables["ml_f_prev_EM"]          = ml_f_prev_EM
                save_variables["shape_seq"]             = shape_S
                save_variables["run_times"]             = [opt_time, mcmc_time]

                pickle_file                 = open(pickle_filename_s, 'wb')
                pickle_output               = pickle.dump(save_variables, pickle_file)
                pickle_file.close()

            n_samples                       = self.__sustainData.getNumSamples()

            if plot:

                #order of subtypes displayed in positional variance diagrams plotted by _plot_sustain_model
                self._plot_subtype_order        = np.argsort(ml_f_EM)[::-1]
                #order of biomarkers in each subtypes' positional variance diagram
                flatten_S = np.vstack([self._flatten_S_dict(s) for s in ml_sequence_EM])
                self._plot_biomarker_order      = flatten_S[self._plot_subtype_order[0], :].astype(int)

            # plot results
            if plot:
                figs, ax = self._plot_sustain_model(ml_sequence_EM=ml_sequence_EM, 
                    samples_sequence=samples_sequence,
                    samples_f=samples_f,
                    n_samples=n_samples,
                    biomarker_labels=self.biomarker_labels,
                    subtype_order=self._plot_subtype_order,
                    biomarker_order=self._plot_biomarker_order,
                    save_path=Path(self.output_folder) / f"{self.dataset_name}_subtype{s+1}_PVD.{plot_format}",
                    figsize=(16,16),
                    **kwargs
                )
                for fig in figs:
                    fig.show()

                ax0.plot(range(self.N_iterations_MCMC), samples_likelihood, label="Subtype " + str(s+1))

        # save and show this figure after all subtypes have been calculcated
        if plot:
            ax0.legend(loc='upper right')
            fig0.tight_layout()
            fig0.savefig(Path(self.output_folder) / f"MCMC_likelihoods.{plot_format}", bbox_inches='tight', dpi=300)
            fig0.show()

        return samples_sequence, samples_f, ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype_stage

    def subtype_and_stage_individuals_newData(self, L_yes_new, L_no_new, num_stages, samples_sequence, samples_f, N_samples):

        sustainData_newData               = sEBMSustainData(L_yes_new, L_no_new, num_stages)
        ml_subtype,         \
        prob_ml_subtype,    \
        ml_stage,           \
        prob_ml_stage,      \
        prob_subtype,       \
        prob_stage,         \
        prob_subtype_stage,_          = self.subtype_and_stage_individuals(sustainData_newData, samples_sequence, samples_f, 100)

        return ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage, prob_subtype, prob_stage, prob_subtype_stage
