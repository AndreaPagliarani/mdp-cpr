# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:52:58 2018

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

def __extract_users_history(dataset_dir, min_user_hops, debug):
    __print_msg('Loading users history...', debug)
    user_history = pd.read_csv(dataset_dir + "user_history.tsv", delimiter = '\t', dtype={"UserID": int, "WindowID": int, "Split": str, "Sequence": int, "JobTitle": str})
    user_history.drop(['WindowID', 'Split'], axis=1, inplace=True)
    user_history.dropna(inplace=True)
    groups = user_history.sort_values(['UserID', 'Sequence'], ascending=True).groupby('UserID')
    users_past_jobs = {}
    for name, group in groups:
        users_past_jobs[name] = group['JobTitle'].tolist()
    for k, v in list(users_past_jobs.items()):
        if len(v) < 2:
            del users_past_jobs[k]
        elif len(set(v)) < min_user_hops:
            del users_past_jobs[k]
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

def evaluate_metrics(pathways, debug):
    """
    Evaluation of the recommended career pathways.
    """
    __print_msg('Evaluating metrics...', debug)
    metrics = {}
    metrics['CareerGoalReached'], metrics['ShorterRecommendedPath'], metrics['UserPathAvgLength'], metrics['RecPathAvgLength'] = metric_path_length(pathways)
    __print_msg('Career goal reached: {}'.format(metrics['CareerGoalReached']), debug)
    __print_msg('Recommended path length: {}'.format(metrics['ShorterRecommendedPath']), debug)
    __print_msg('User pathway average length: {}'.format(metrics['UserPathAvgLength']), debug)
    __print_msg('Recommended pathway average length: {}'.format(metrics['RecPathAvgLength']), debug)

def metric_path_length(pathways):
    """
    Compute the percentage of times the recommended pathway is shorter than the user's one.
    """
    num_users = len(pathways)
    num_good_recommendations = 0
    sum_u_path_len = 0
    sum_r_path_len = 0
    career_goal_reached = 0
    for user, pathway_tuple in pathways.items():
        u_path = pathway_tuple[0]
        r_path = pathway_tuple[1]
        sum_u_path_len += len(u_path)
        sum_r_path_len += len(r_path)
        if r_path[-1]==u_path[-1]:
            career_goal_reached += 1
            if len(r_path) < len(u_path):
                num_good_recommendations += 1
    return 100.0 * career_goal_reached/num_users, 100.0 * num_good_recommendations / num_users, sum_u_path_len/num_users, sum_r_path_len/num_users 
    
def run_single_exp(params, debug=True):
    exp_start_time = time.time()
    
    # loading job postings
    start_time = exp_start_time
    job_postings = __load_job_postings(params['dataset_dir'], params['job_postings_fraction'], debug)
    end_time = time.time()
    __print_msg('Job postings loaded in {} seconds'.format(end_time - start_time), debug)
    
    # building a job graph
    job_graph = JobGraph(job_postings, 'Text', num_clusters=params['num_clusters'], max_skillset_size=params['skills_per_cluster'], debug=debug)
    
    # select users
    start_time = time.time()
    users_history = __extract_users_history(params['dataset_dir'], params['min_user_hops'], debug)
    end_time = time.time()
    __print_msg('Users history extracted in {} seconds'.format(end_time - start_time), debug)
    
    # compute pathways
    start_time = end_time
    pathways = compute_pathways(users=users_history, job_graph=job_graph, debug=debug)
    end_time = time.time()
    __print_msg('Pathways computed in {} seconds'.format(end_time - start_time), debug)
    
    # evaluate metrics
    start_time = end_time
    evaluate_metrics(pathways, debug)
    end_time = time.time()
    __print_msg('Metrics evaluated in {} seconds'.format(end_time - start_time), debug)
    
    exp_end_time = time.time()
    __print_msg('Experiment total duration: {} seconds'.format(exp_end_time - exp_start_time), debug)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-jpf", "--job_postings_fraction", type=float, help="the fraction of job postings used to learn a text representation as well as to build a job graph (default 0.01)")
parser.add_argument("-nc", "--num_clusters", type=int, help="the number of clusters used to group job postings as well as the number of job graph states (default 20)")
parser.add_argument("-spc", "--skills_per_cluster", type=int, help="the number of representative terms for each cluster (default 100)")
parser.add_argument("-muh", "--min_user_hops", type=int, help="the minimum number of career hops for an user (default 10)")
parser.add_argument("-ddir", "--dataset_dir", type=str, help="the dataset path (default current folder)")
parser.add_argument("-sdir", "--script_dir", type=str, help="the script directory (default current folder)")

import time
import pandas as pd
import os

if __name__ == "__main__":
    args = parser.parse_args()
    params = {}
    params['job_postings_fraction'] = args.job_postings_fraction if args.job_postings_fraction else 0.01
    params['num_clusters'] = args.num_clusters if args.num_clusters else 20
    params['skills_per_cluster'] = args.skills_per_cluster if args.skills_per_cluster else 100
    params['min_user_hops'] = args.min_user_hops if args.min_user_hops else 10
    params['dataset_dir'] = args.dataset_dir if args.dataset_dir else os.path.dirname(os.path.realpath(__file__))
    script_dir = args.script_dir if args.script_dir else os.path.dirname(os.path.realpath(__file__))
    import sys
    sys.path.append(script_dir)
    from job_graph import JobGraph
    from mdp import MDP
    run_single_exp(params)