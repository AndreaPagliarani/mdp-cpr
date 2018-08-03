# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:56:48 2018

@author: andreap
"""
#%%
import random as rd
import numpy as np
script_dir = 'C:/Users/IBM_ADMIN/Desktop/MyCa/gitRepo/careerpathrecommendation/code/python/cpr_method'
import sys
sys.path.append(script_dir)
from utility import Preprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import MiniBatchKMeans
import time
import math

class JobGraph:
    """
    This class allows to create a job graph starting from data (e.g. clustering of jobs).
    Each state will be represented by a skillset.
    """
    def __init__(self, data, text_column, min_instances_per_cluster=None, num_clusters=None, max_skillset_size=100, seed=1000, debug=False):
        """
        - data is the input DataFrame to be used for building the graph
        - text_column is the DataFrame column containing the text where to extract skills from
        - min_instances_per_cluster If it is not None, the clusters containing less than the specified instances will be pruned. If a number x between 0 and 1 is given, the clusters containing less instances than x*num_instances/num_clusters will be pruned. 
        - num_clusters is the number of clusters to be used. If None, it will be inferred from data (computationally more expensive, because several attempts have to be done).
        - max_skillset_size is the maximum number of terms representing each cluster/instance
        """
        start_time = time.time()
        
        rd.seed(seed)
        self.prep = Preprocessing()
        self.__debug = debug
        self.max_skillset_size = max_skillset_size
        self.data = data
        self.__text_column = text_column
        
        self.__preprocess_data()
        
        end_time = time.time()
        print('Text pre-processed in {} seconds'.format(end_time - start_time))
        start_time = end_time
        
        self.text_repr_model, self.text_repr = self.__get_text_representation()
        
        end_time = time.time()
        print('Text representation learnt in {} seconds'.format(end_time - start_time))
        start_time = end_time
        
        self.clustering_model, self.clusters_pruned = self.__clustering(k=num_clusters, min_instances_per_cluster=min_instances_per_cluster)
        self.__centroids = {i: c for i, c in enumerate(self.clustering_model.cluster_centers_)}
        
        end_time = time.time()
        print('Data clustered in {} seconds'.format(end_time - start_time))
        start_time = end_time
                
        self.skillsets = self.__extract_skillsets_from_clusters(max_skillset_size)
        
        end_time = time.time()
        print('Cluster skillsets extracted in {} seconds'.format(end_time - start_time))
        start_time = end_time
                
        self.job_graph = self.__building_job_graph()
        
        end_time = time.time()
        print('Job graph built in {} seconds'.format(end_time - start_time))
        
    def __print_msg(self, msg):
        if self.__debug is True:
            print(msg)
    
    def __preprocess_data(self):
        """
        Pre-process text data.
        """
        self.__print_msg('Text pre-processing...')
        self.data[self.__text_column] = self.data[self.__text_column].apply(lambda x: self.prep.preprocess_data(x, stopwords_removal=False))
        self.data.dropna(inplace=True)
    
    def __get_text_representation(self, method='d2v'):
        """
        Convert text_column into a suitable representation. Currently, only 'd2v' option is available, but 'tfidf' will be implemented soon.
        """
        if method == 'd2v':
            self.__print_msg('Learning Doc2Vec representation...')
            docs = [TaggedDocument(words=d, tags=[str(i)]) for i, d in enumerate(self.data.iloc[:][self.__text_column])]
            model = self.__train_d2v(docs, alpha=0.025, min_alpha=0.001, passes=30)
            text_repr = model.docvecs.vectors_docs
        else:
            raise Exception('Unsupported word representation!')
        return model, text_repr
        
    def __train_d2v(self, docs, alpha, min_alpha, passes):
        """
        Train a Doc2Vec model using the docs given as input.
        """
        copy_docs = docs.copy()
        alpha_delta = (alpha - min_alpha) / (passes - 1)
        d2v_model = Doc2Vec(dm=0, vector_size=100, window=5, negative=5, min_count=5, dbow_words=1, workers=4)
        d2v_model.build_vocab(copy_docs)
        for epoch in range(passes):
            rd.shuffle(copy_docs)
            d2v_model.alpha, d2v_model.min_alpha = alpha, alpha
            d2v_model.train(copy_docs, total_examples=d2v_model.corpus_count, epochs=1)
            alpha -= alpha_delta
        return d2v_model
        
    def __clustering(self, k, min_instances_per_cluster, algorithm='MiniBatchKMeans'):
        """
        Apply a clustering algorithm to group data. Currently, only 'MiniBatchKMeans' algorithm is available.
        If k is None, the best number of clusters will be automatically inferred from data, looking at clustering inertia.
        """
        if algorithm == 'MiniBatchKMeans':
            self.__print_msg('Applying MiniBatchKMeans...')
            if k is None:
                k = self.__find_best_k(max_k_allowed=0.1*len(self.data))
            self.__print_msg('Clustering...')
            model = MiniBatchKMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1, init_size=1000, batch_size=1000)
            model.fit(self.text_repr)
            self.__add_clusters_to_data(model)
            clusters_pruned = self.__prune_clusters(model, min_instances_per_cluster)
            return model, clusters_pruned
        else:
            raise Exception('Unsupported algorithm')
    
    def __find_best_k(self, max_k_allowed):
        """
        Process 10 possible values of k. 
        Stop when the average of these 10 values is not at least 5% better than the average of the previous 10.
        When stopping, return the best k among the previous 10.
        The minimum allowed value for max_k_allowed is 10.
        """
        self.__print_msg('Tuning the number of clusters on data...')
        max_k_allowed = 10 if max_k_allowed < 10 else int(max_k_allowed)
        possible_k = list(range(1, max_k_allowed+1))
        inertia = {}
        avg_prev_inertia = sys.maxsize
        for k in possible_k:
            cluster_model = MiniBatchKMeans(n_clusters=k, init="k-means++", max_iter=100, n_init=1, init_size=1000, batch_size=1000)
            cluster_model.fit(self.text_repr)
            inertia[k] = cluster_model.inertia_
            if k % 10 == 0:
                # check stop condition
                avg_inertia = sum(inertia.values())/len(inertia)
                if 0.95*avg_prev_inertia < avg_inertia:
                    break
                avg_prev_inertia = avg_inertia
                previous_inertia = inertia
                inertia = {}
            # stop condition: last 3 values     
        return min(previous_inertia, key=previous_inertia.get)
        
    def __prune_clusters(self, model, min_instances):
        """
        Prune clusters having less than min_instances.
        If a number x between 0 and 1 is given, the clusters containing less instances than 
        x*num_instances/num_clusters will be pruned.
        """
        self.__print_msg('Pruning small clusters...')
        clusters = np.asarray(np.unique(model.labels_.tolist(), return_counts=True)).T
        thr = 0.1*sum(clusters[:,1])/len(clusters) if min_instances is None else min_instances if min_instances > 1 else min_instances*sum(clusters[:,1])/len(clusters)
        clusters_pruned = set()
        for c in clusters:
            if c[1] < thr:
                clusters_pruned.add(c[0])
        return clusters_pruned
    
    def __add_clusters_to_data(self, clustering_model, cluster_col='Cluster'):
        """
        Assign clusters to data. Each instance will be assigned to exactly one cluster.
        """
        self.data[cluster_col] = 0
        self.data[cluster_col] = self.data[cluster_col].astype(np.int16)
        labels = clustering_model.labels_.tolist()
        for i in range(len(self.data)):
            self.data.iloc[i, self.data.columns.get_loc(cluster_col)] = labels[i]
    
    def __extract_skillsets_from_clusters(self, max_skillset_size):
        """
        Given a clustering model, rank words in each cluster, and return the list of ranked words.
        The most relevant max_skillset_size words of each cluster are the skills characterizing it.
        """
        self.__print_msg('Building skillsets from clusters...')
        centroids = {i: c for i, c in enumerate(self.clustering_model.cluster_centers_) if i not in self.clusters_pruned}
        skillsets = {}
        for i, c in centroids.items():
            best_terms_and_scores = self.text_repr_model.wv.most_similar(positive=[c], topn=max_skillset_size)
            skillsets[i] = [x[0] for x in best_terms_and_scores]
        return skillsets
    
    def __building_job_graph(self):
        """
        Build a job graph, starting from the skillsets extracted from clusters.
        """
        self.__print_msg('Building the job graph...')
        jg = dict()
        for i in self.skillsets:
            connections = ()
            for j in self.skillsets:
                prob = self.__transition_likelihood(source_node=i, dest_node=j)
                connections = connections + ((j, prob),)
            jg[i] = connections
        # sort connections by decreasing likelihood
        for node, conn in jg.items():
            jg[node] = sorted(conn, key=lambda tup: tup[1], reverse=True)
        return jg
    
    def __transition_likelihood(self, dest_node, source_node=None, user_skills=None, exponential=True):
        """Computes the probability of moving from the source node to the destination node.
        If both source_node and user_skills are given, the latter will be ignored.
        """
        if source_node is None and user_skills is None:
            raise Exception("Either source_node or user_skills must be given as input.")
        source_skills = set(self.skillsets[source_node]) if source_node is not None else set(user_skills)
        dest_skills = set(self.skillsets[dest_node])
        source_dest_skills = source_skills & dest_skills
        likelihood = len(source_dest_skills) / len(dest_skills)
        if exponential is True:
            likelihood = (math.exp(likelihood) - 1) / (math.exp(1) - 1)
        return likelihood
    
    def assign_job_to_jobgraph_state(self, job):
        """
        Map the job given as input to the most similar state of the job graph, and retrieve the corresponding skillset.
        """
        job_p = self.prep.preprocess_data(job, stopwords_removal=False)
        vector_instance = self.text_repr_model.infer_vector(doc_words=job_p, steps=1000, alpha=0.025)
        assigned = False
        while assigned is False:
            assigned_cluster = self.clustering_model.predict(np.reshape(vector_instance, newshape=(1, -1)))[0]
            try:
                assigned_skillset = self.skillsets[assigned_cluster]
                assigned = True
            except:
                """The instance has been assigned to a pruned cluster. Expand the instance with the skillset characterising that cluster, then try to reassign it."""
                terms_and_scores = self.text_repr_model.wv.most_similar(positive=[self.__centroids[assigned_cluster]], topn=self.max_skillset_size)
                job_p = job_p + [x[0] for x in terms_and_scores]
        return assigned_cluster, assigned_skillset
#%%
def test_Job_Graph():
    import pandas as pd
    dataset_dir = "C:/Users/IBM_ADMIN/.kaggle/competitions/job-recommendation/"
    job_postings = pd.read_csv(dataset_dir + "jobs.tsv", skiprows=[122432,602575,990949], delimiter = '\t', dtype={"JobID": int, "WindowID": int, "Title": str, "Description": str, "Requirements": str, "City": str, "State": str, "Country": str, "Zip5": str, "StartDate": str, "EndDate": str})
    job_postings.drop(['WindowID', 'City', 'State', 'Country', 'Zip5', 'StartDate', 'EndDate'], axis=1, inplace=True)
    job_postings = job_postings.sample(frac=0.001)
    job_postings['Text'] = job_postings['Title'] + ' ' + job_postings['Description'] + ' ' + job_postings['Requirements']
    job_postings.drop(['Title', 'Description', 'Requirements'], axis=1, inplace=True)
    job_postings.dropna(inplace=True)
    return JobGraph(job_postings, 'Text', num_clusters=20, debug=True)