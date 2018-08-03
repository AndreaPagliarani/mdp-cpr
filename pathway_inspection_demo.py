# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:56:08 2018

@author: andreap
"""

def __print_msg(msg, debug):
    if debug is True:
        print(msg)
        
def __load_job_postings(dataset_dir, fraction, debug):
    __print_msg('Loading job postings...', debug)
    job_postings = pd.read_csv(dataset_dir + "jobs.tsv", skiprows=[122432,602575,990949], delimiter = '\t', dtype={"JobID": int, "WindowID": int, "Title": str, "Description": str, "Requirements": str, "City": str, "State": str, "Country": str, "Zip5": str, "StartDate": str, "EndDate": str})
    job_postings.drop(['WindowID', 'City', 'State', 'Country', 'Zip5', 'StartDate', 'EndDate'], axis=1, inplace=True)
    if fraction < 1:
        job_postings = job_postings.sample(frac=fraction)
    job_postings['Text'] = job_postings['Title'] + ' ' + job_postings['Description'] + ' ' + job_postings['Requirements']
    job_postings.drop(['Title', 'Description', 'Requirements'], axis=1, inplace=True)
    job_postings.dropna(inplace=True)
    return job_postings

def __extract_users_history(dataset_dir, user_id, max_user, debug):
    __print_msg('Loading users history...', debug)
    user_history = pd.read_csv(dataset_dir + "user_history.tsv", delimiter = '\t', dtype={"UserID": int, "WindowID": int, "Split": str, "Sequence": int, "JobTitle": str})
    user_history.drop(['WindowID', 'Split'], axis=1, inplace=True)
    user_history.dropna(inplace=True)
    groups = user_history.sort_values(['UserID', 'Sequence'], ascending=True).groupby('UserID')
    users_past_jobs = {}
    for name, group in groups:
        if user_id is None:
            users_past_jobs[name] = group['JobTitle'].tolist()
        elif name == user_id:
            users_past_jobs[name] = group['JobTitle'].tolist()
            break
    num_users = 0
    for k, v in list(users_past_jobs.items()):
        if len(v) < 10 or num_users == max_user:
            del users_past_jobs[k]
        else:
            num_users += 1
    return users_past_jobs
    
def compute_pathways(users, job_graph, debug, min_likelihood_thr=0.2):
    """
    Compute career pathways (i.e. user-performed and recommended) for each user.
    Return type: 
        userID : (user_pathway, recommended_pathway)
    """
    start_time = time.time()
    __print_msg('Computing career pathways...', debug)
    user_pathways = {}
    tot_users = len(users)
    i = 0
    for user, user_jobs in users.items():
        user_pathway = compute_user_pathway(user_jobs, job_graph)
        recommended_pathway = recommend_pathway(user_jobs, job_graph, user_pathway[-1], min_likelihood_thr)
        user_pathways[user] = (user_pathway, recommended_pathway)
        i += 1
        if i % 1000 == 0:
            __print_msg('Num users processed: {}/{}'.format(i, tot_users), debug)
            end_time = time.time()
            __print_msg('Execution time: {} seconds'.format(end_time - start_time), debug)
    return user_pathways
    
def compute_user_pathway(user_jobs, job_graph, debug=False):
    """
    Compute the user pathway, given the sequence of job titles.
    The first job will be excluded, because it is only used to build the user's initial profile.
    """
    pathway = []
    for i, job in enumerate(user_jobs):
        if i == 0:
            continue
        cluster, _ = job_graph.assign_job_to_jobgraph_state(job)
        pathway.append(cluster)
    return pathway

def recommend_pathway(user_jobs, job_graph, goal_state, min_likelihood_thr):
    """
    Recommend a pathway, given the sequence of job titles.
    """
    user_jobs_for_mdp = [user_jobs[0]]
    mdp = MDP(job_graph, user_jobs_for_mdp, goal_state, min_likelihood_thr=min_likelihood_thr)
    return mdp.solve_mdp()
    
def compute_pathway_for_user(params, debug=True):
    exp_start_time = time.time()
    
    # select users
    start_time = exp_start_time
    users_history = __extract_users_history(params['dataset_dir'], params['user_id'], params['max_user'], debug)
    end_time = time.time()
    if not users_history:
        __print_msg('User {} does not exist'.format(params['user_id']), debug)
        return
    if params['user_id'] is not None:
        __print_msg('History of user {} extracted in {} seconds'.format(params['user_id'], end_time - start_time), debug)
    else:
        __print_msg('Users history of {} users extracted in {} seconds'.format(len(users_history), end_time - start_time), debug)
    
    # loading job postings
    start_time = end_time
    job_postings = __load_job_postings(params['dataset_dir'], params['job_postings_fraction'], debug)
    end_time = time.time()
    __print_msg('Job postings loaded in {} seconds'.format(end_time - start_time), debug)
    
    # building a job graph
    job_graph = JobGraph(job_postings, 'Text', num_clusters=params['num_clusters'], max_skillset_size=params['skills_per_cluster'], debug=debug)
    
    # compute pathways
    start_time = time.time()
    pathways = compute_pathways(users=users_history, job_graph=job_graph, debug=debug)
    end_time = time.time()
    __print_msg('Pathways computed in {} seconds'.format(end_time - start_time), debug)
    
    # output pathways
    for user, user_jobs in users_history.items():
        __print_msg('Job sequence of user {}: {}'.format(user, user_jobs), debug)
        pathway_tuple = pathways[user]
        __print_msg('Pathway of user {}: {}'.format(user, pathway_tuple[0]), debug)
        __print_msg('Recommended pathway to user {}: {}'.format(user, pathway_tuple[1]), debug)
        __print_msg('', debug)
            
    
    exp_end_time = end_time
    __print_msg('Experiment total duration: {} seconds'.format(exp_end_time - exp_start_time), debug)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-uid", "--user_id", type=int, help="the user ID the pathway will be computed for (default None); if None, max_user users will be displayed")
parser.add_argument("-mu", "--max_user", type=int, help="the maximum number of users to be shown (default 10)")
parser.add_argument("-ddir", "--dataset_dir", type=str, help="the dataset path (default current folder)")
parser.add_argument("-sdir", "--script_dir", type=str, help="the script directory (default current folder)")

import time
import pandas as pd
import os

if __name__ == "__main__":
    args = parser.parse_args()
    params = {}
    params['job_postings_fraction'] = 0.01
    params['num_clusters'] = 40
    params['skills_per_cluster'] = 100
    params['user_id'] = args.user_id if args.user_id else None
    params['max_user'] = args.max_user if args.max_user else 10
    params['dataset_dir'] = args.dataset_dir if args.dataset_dir else os.path.dirname(os.path.realpath(__file__))
    script_dir = args.script_dir if args.script_dir else os.path.dirname(os.path.realpath(__file__))
    import sys
    sys.path.append(script_dir)
    from job_graph import JobGraph
    from mdp import MDP
    compute_pathway_for_user(params)