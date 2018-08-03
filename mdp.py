# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:56:48 2018

@author: andreap
"""
#%%
import time
import numpy as np
from collections import Counter
import math
from scipy.sparse import csr_matrix, lil_matrix
from mdptoolbox.mdp import PolicyIteration, PolicyIterationModified, ValueIteration
import random as rd

class MDP:
    """
    This class allows creating a Markov Decision Process, starting from a JobGraph, a list of preceding job positions and a career goal.
    The MDP can then be solved by means of the Policy Iteration algorithm.
    """
    def __init__(self, job_graph, user_job_history, user_career_goal, min_likelihood_thr, debug=False):
        """
        min_likelihood_thr is a threshold that forces the user to apply only to jobs if the likelihood to succeed is higher than this threshold. This avoids considering most unlikely user behavior, and allows saving computational time. 
        """
        self.__debug = debug
        self.job_graph = job_graph
        
        start_time = time.time()
        
        self.__print_msg('Mapping the career goal to the goal state...')
        try:
            self.goal_state = int(user_career_goal)
            self.goal_skillset = self.job_graph.skillsets[self.goal_state]
        except:
            self.goal_state, self.goal_skillset = self.job_graph.assign_job_to_jobgraph_state(user_career_goal)
        
        end_time = time.time()
        self.__print_msg('Execution time: {} seconds'.format(end_time - start_time))
        start_time = end_time
        
        self.user_initial_profile = self.__build_user_profile(user_job_history)
        
        end_time = time.time()
        self.__print_msg('Execution time: {} seconds'.format(end_time - start_time))
        start_time = end_time
                
        self.transitions, self.rewards = self.__create_MDP(min_likelihood_thr)
        
        end_time = time.time()
        self.__print_msg('Execution time: {} seconds'.format(end_time - start_time))
    
    def solve_mdp(self, algorithm='PolicyIteration', discount=0.999):
        """
        Run the algorithm over the Markov Decision Process built.
        Available algorithms: PolicyIteration, PolicyIterationModified, ValueIteration (default).
        """
        self.__print_msg('Solving MDP...')
        alg_setup = PolicyIteration(self.transitions, self.rewards, discount=discount) if algorithm=='PolicyIteration' else PolicyIterationModified(self.transitions, self.rewards, discount=discount) if algorithm=='PolicyIterationModified' else ValueIteration(self.transitions, self.rewards, discount=discount)
        alg_setup.run()
        optimal_policy = [self.jg_actions[i] for i in alg_setup.policy]
        try:
            goal_index = optimal_policy.index(self.goal_state) + 1
        except:
            goal_index = None
        return optimal_policy[:goal_index]
    
    def __create_MDP(self, min_likelihood_thr, negative_reward=-0.2, init_state_id=-100):
        """
        Map the job graph, the user's profile and the career goal into a Markov Decision Process, whose solution is suitable to recommend a career pathway to the user.
        """
        transitions = []
        rewards = []
        init_state = set()
        init_state.add(init_state_id)
        mdp_states_dict = {0 : init_state}
        mdp_states_dict_accumulator = mdp_states_dict.copy()
        count_mdp_states = 1
        self.__print_msg('Building MDP career states...')
        mdp_state_added = True
        while mdp_state_added is True: #cycle until new MDP states are added
            mdp_state_added = False
            new_mdp_states_dict = {}
            for index, mdp_state in mdp_states_dict.items(): #compute new reachable states
                cur_profile = self.__get_cur_profile(mdp_state, init_state_id)
                #self.__print_msg(mdp_state)
                #self.__print_msg(cur_profile)
                #self.__print_msg('')
                success_probs, pos_rewards = self.__compute_successful_application_probabilities_and_reward(mdp_state, cur_profile, self.job_graph.skillsets, min_likelihood_thr)
                if any(success_probs) is True: #new reachable states found
                    mdp_state_added = True
                    for mdp_allowed_action, prob in success_probs.items():
                        new_mdp_state = mdp_state.copy()
                        new_mdp_state.add(mdp_allowed_action)
                        #self.__print_msg('old state: {}'.format(mdp_state))
                        #self.__print_msg('action: {}'.format(mdp_allowed_action))
                        #self.__print_msg('new state: {}'.format(new_mdp_state))
                        #self.__print_msg('')
                        mdp_state_as_dict_entry = {index: mdp_state}
                        if new_mdp_state not in new_mdp_states_dict.values(): #new MDP state
                            new_mdp_states_dict[count_mdp_states] = new_mdp_state
                            new_mdp_state_as_dict_entry = {count_mdp_states: new_mdp_state}
                            count_mdp_states += 1
                        else: #already existing MDP state
                            for k, v in new_mdp_states_dict.items():
                                if v==new_mdp_state:
                                    new_mdp_state_as_dict_entry = {k: v}
                                    break
                        transitions.append(tuple((mdp_state_as_dict_entry, new_mdp_state_as_dict_entry, mdp_allowed_action, prob)))
                        transitions.append(tuple((mdp_state_as_dict_entry, mdp_state_as_dict_entry, mdp_allowed_action, 1-prob)))
                        rewards.append(tuple((mdp_state_as_dict_entry, new_mdp_state_as_dict_entry, mdp_allowed_action, pos_rewards[mdp_allowed_action])))
                        rewards.append(tuple((mdp_state_as_dict_entry, mdp_state_as_dict_entry, mdp_allowed_action, negative_reward)))
                        
            mdp_states_dict = new_mdp_states_dict
            mdp_states_dict_accumulator.update(new_mdp_states_dict)
        self.__print_msg('MDP career states found: {}'.format(count_mdp_states))
        self.__print_msg('Transitions: {} - Rewards: {}'.format(len(transitions),len(rewards)))
        self.__print_msg('Building the MDP...')
        self.jg_actions = list(self.job_graph.skillsets.keys())
        mdp_transitions = []
        mdp_rewards = []
        for _ in range(len(self.jg_actions)): #init lil matrix
            mdp_transitions.append(lil_matrix((count_mdp_states,count_mdp_states)))
            mdp_rewards.append(lil_matrix((count_mdp_states,count_mdp_states)))
        for i, tr in enumerate(transitions): #fill-in transitions
            tr_action_index = self.jg_actions.index(tr[2])
            tr_source_state_index = list(tr[0].keys())[0]
            tr_dest_state_index = list(tr[1].keys())[0]
            tr_prob = tr[3]
            #self.__print_msg('')
            #self.__print_msg(tr_action_index)
            #self.__print_msg(tr_source_state_index)
            #self.__print_msg(tr_dest_state_index)
            #self.__print_msg(tr_prob)
            mdp_transitions[tr_action_index][tr_source_state_index,tr_dest_state_index] = tr_prob
            rew = rewards[i]
            rew_action_index = self.jg_actions.index(rew[2])
            rew_source_state_index = list(rew[0].keys())[0]
            rew_dest_state_index = list(rew[1].keys())[0]
            rew_score = rew[3]
            #self.__print_msg('')
            #self.__print_msg(rew_action_index)
            #self.__print_msg(rew_source_state_index)
            #self.__print_msg(rew_dest_state_index)
            #self.__print_msg(rew_score)
            mdp_rewards[rew_action_index][rew_source_state_index,rew_dest_state_index] = rew_score
        mdp_transitions_csr = mdp_transitions.copy()
        avoid_transition_reward = negative_reward * 100
        for i in range(len(self.jg_actions)): #convert to csr matrix
            mdp_transitions_csr[i] = mdp_transitions_csr[i].tocsr()
            for row, has_any_element in enumerate(np.diff(mdp_transitions_csr[i].indptr) != 0):
                if bool(has_any_element) is False:
                    locations_to_pad = self.__generate_random_location_to_pad(count_mdp_states, min(count_mdp_states, 10))
                    for index, val in locations_to_pad.items():
                        mdp_transitions[i][row, index] = val
                        mdp_rewards[i][row, index] = avoid_transition_reward
            try:
                np.linalg.inv(mdp_transitions[i].todense())
            except:
                self.__print_msg(mdp_transitions[i].todense())
                return
        for i in range(len(self.jg_actions)): #convert to csr matrix
            mdp_transitions[i] = mdp_transitions[i].tocsr()
            mdp_rewards[i] = mdp_rewards[i].tocsr()
        return mdp_transitions, mdp_rewards
    
    def __generate_random_location_to_pad(self, max_state, num_loc):
        """
        Pad num_loc random locations to random numbers, in order for the MDP transition matrices to be invertible.
        """
        loc = {}
        indexes = rd.sample(range(0, max_state), num_loc)
        for i, value in enumerate(np.random.dirichlet(np.ones(num_loc), size=1)[0]):
            if i != num_loc-1:
                loc[indexes[i]] = value
            else:
                loc[indexes[i]] = 1 - sum(loc.values())
        return loc
    
    def __get_cur_profile(self, mdp_state, init_state_id):
        """
        Get the current user profile when the user is in mdp_state.
        """
        profile = self.user_initial_profile.copy()
        mdp_state_copy = mdp_state.copy()
        mdp_state_copy.remove(init_state_id)
        for jp_state in mdp_state_copy:
            profile.update(set(self.job_graph.skillsets[jp_state]))
        return profile
    
    def __compute_successful_application_probabilities_and_reward(self, mdp_cur_state, profile, skillsets, min_likelihood_thr, M = 10):
        """
        For each state, compute:
            - the probability that, if the user applied for a job, her application would succeed, cutting off those lower than min_likelihood_thr;
            - the reward that, if the user applied for a job, she would get in case of success.
        """
        likelihoods = {}
        rewards = {}
        if self.goal_state in mdp_cur_state: #the current state is a goal state
            return likelihoods, rewards
        jg_skillsets = skillsets.copy()
        for jg_state in mdp_cur_state:
            jg_skillsets.pop(jg_state, None)
        user_skills = set(profile.keys())
        goal_skills = set(self.goal_skillset)
        missing_skills = goal_skills - user_skills
        for jg_state, jg_skillset in jg_skillsets.items():
            dest_skills = set(jg_skillset)
            source_dest_skills = user_skills & dest_skills
            overlapping_factor = len(source_dest_skills) / len(dest_skills)
            likelihood = (math.exp(overlapping_factor) - 1) / (math.exp(1) - 1)
            if likelihood > min_likelihood_thr:
                likelihoods[jg_state] = likelihood
                missing_skills_after_dest_node = missing_skills - dest_skills
                rewards[jg_state] = M if jg_state==self.goal_state else 1 - len(missing_skills_after_dest_node)/len(missing_skills) if len(missing_skills) > 0 else 0
        return likelihoods, rewards
    
    def __print_msg(self, msg):
        if self.__debug is True:
            print(msg)
    
    def __build_user_profile(self, job_history):
        """
        Build the user's profile, given her job history.
        """
        self.__print_msg('Building user profile...')
        profile = []
        for job in job_history:
            _, skillset = self.job_graph.assign_job_to_jobgraph_state(job)
            profile.extend(skillset)
        return Counter(profile)
    """
    def __assign_job_to_jobgraph_state(self, job):
        
        job_p = self.job_graph.prep.preprocess_data(job, stopwords_removal=False)
        vector_instance = self.job_graph.text_repr_model.infer_vector(doc_words=job_p, steps=1000, alpha=0.025)
        assigned = False
        while assigned is False:
            assigned_cluster = self.job_graph.clustering_model.predict(np.reshape(vector_instance, newshape=(1, -1)))[0]
            try:
                assigned_skillset = self.job_graph.skillsets[assigned_cluster]
                assigned = True
            except:
                
                terms_and_scores = self.job_graph.text_repr_model.wv.most_similar(positive=[self.__centroids[assigned_cluster]], topn=20)
                job_p = job_p + [x[0] for x in terms_and_scores]
        return assigned_cluster, assigned_skillset
    """
#%%
def test_MDP():
    # A JobGraph should already exist
    import pandas as pd
    dataset_dir = "C:/Users/IBM_ADMIN/.kaggle/competitions/job-recommendation/"
    user_history = pd.read_csv(dataset_dir + "user_history.tsv", delimiter = '\t', dtype={"UserID": int, "WindowID": int, "Split": str, "Sequence": int, "JobTitle": str})
    user_history.drop(['WindowID', 'Split'], axis=1, inplace=True)
    user_history.dropna(inplace=True)
    users_past_jobs = extract_users_history(user_history, num_users=1)
    user_jobs = list(users_past_jobs.values())[0]
    career_goal = "financial analysis finance accounting analyst reporting business budget management report senior process planning monthly analyze forecast provide company data cost"
    return MDP(jg, user_jobs, career_goal, min_likelihood_thr=0.2, debug=True)
    
def extract_users_history(data, num_users):
    groups = data.sort_values(['UserID', 'Sequence'], ascending=True).groupby('UserID')
    users_past_jobs = {}
    i = 0
    for name, group in groups:
        users_past_jobs[name] = group['JobTitle'].tolist()
        i += 1
        if i == num_users:
            break
    for k, v in list(users_past_jobs.items()):
        if len(v) < 2:
            del users_past_jobs[k]
    return users_past_jobs

def compute_reward(goal_skillset, dest_skillset, user_profile=None, source_skillset=None):
    if user_profile is not None:
        user_skills = set(user_profile.keys())
    else:
        user_skills = set(source_skillset)
    goal_skills = set(goal_skillset)
    dest_skills = set(dest_skillset)
    missing_skills = goal_skills - user_skills
    missing_skills_after_dest_node = missing_skills - dest_skills
    return 1 - len(missing_skills_after_dest_node)/len(missing_skills) if len(missing_skills) > 0 else 0

def compute_likelihood(dest_skillset, user_profile=None, source_skillset=None):
    if user_profile is not None:
        user_skills = set(user_profile.keys())
    else:
        user_skills = set(source_skillset)
    dest_skills = set(dest_skillset)
    source_dest_skills = user_skills & dest_skills
    overlapping_factor = len(source_dest_skills) / len(dest_skills)
    return (math.exp(overlapping_factor) - 1) / (math.exp(1) - 1)

def compute_likelihood_reward(goal_skillset, dest_skillset, user_profile=None, source_skillset=None):
    likelihood = compute_likelihood(dest_skillset=dest_skillset, user_profile=user_profile, source_skillset=source_skillset)
    reward = compute_reward(goal_skillset=goal_skillset, dest_skillset=dest_skillset, user_profile=user_profile, source_skillset=source_skillset)
    return likelihood, reward