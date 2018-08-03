# A Markov Decision Process based career pathway recommender (mdp-cpr)
This package provides a Python implementation of the career pathway recommender that currently is under peer-review in <a href="https://aaai.org/Conferences/AAAI-19/aaai19call/#">AAAI-2019</a>.

Any publication that discloses findings either arising from this source code or about career pathway recommendation must cite "A skillset-based approach to modelling job transitions and recommending career pathways".

<h2>Introduction</h2>
Career Pathway Recommendation (CPR) is an emerging research thread that aims at identifying a sequence of jobs for a user to reach her career goal. When a candidate is looking for a job, but she still does not have an adequate profile to get it, a job recommender will inherently fail, either hiding the opportunity to the user or leading her towards a failure. On the other hand, a CP recommender will compute a policy that eventually leads the user towards her goal, minimising the required time while simultaneously improving the user's profile and increasing the likelihood that the user application for the career goal succeeds. <br/>

The approach consists of two main steps:
- An user-independent job graph is built from a set of job postings. The job graph has a state for each cluster of related jobs, whereas a state transition from state <i>i</i> to state <i>j</i> represents the likelihood to get a job in state <i>j</i> given that <i>i</i> is the current state.
- An user-dependent recommendation is performed by solving a different MDP for each user. This guarantees the recommended pathway is personalised as well as optimum, since the computed policy takes into account all possible career evolutions.

<h2>Dataset</h2>
As far as we know, a dataset suitable for career pathway recommendation is not publicly available.
In order to perform some preliminar experiments, we relied on a publicly available dataset from <a href="https://www.kaggle.com/c/job-recommendation/data">Kaggle</a>, which has to be downloaded to run these experiments.
Since such data are suitable for job recommendation, so we had to introduce many expedients to actually test the method, which can negatively affect results. <br/>

The current limitations as well as the introduced expedients are the following:
- <b>The user's job history is a sequence of job titles.</b>
Job titles cannot easily be mapped into existing job postings. Each job title is mapped into the most similar cluster.
- <b>The user's profile is unavailable.</b>
Once a job is mapped into a cluster, the most representative terms of such a cluster can be selected as the skillset the user has acquired performing a job within it and interpreted as her initial profile.
- <b>The user's career goal is unknown.</b> The very last job in the user's history is interpreted as her career goal, and the goal state is again selected by mapping this job into the most likely cluster. Anyway it is likely that the very last job the user had is not exactly what she is looking for, since the dataset also contains some applications users made to other jobs at a later time. The user can theoretically be far from reaching her actual career goal, but here we are forced to consider as the career goal has already been reached.
- <b>The job postings representation is achieved via <a href="https://radimrehurek.com/gensim/models/doc2vec.html">Doc2Vec</a>.</b>
Word ordering and semantics are taken into account by such a model, but results are irreproducible because of parallelism. Furthermore, Doc2Vec needs huge datasets to be trained on, even larger than the one we had. As an alternative, we could rely on a pre-trained Word2Vec model for word representation as the one trained on <a href="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit">Google News</a>, but unfortunately only about the 33% of the words in job postings were included into such a model.
- <b>Job duration is missing.</b> Our experiments take into account career hops, but duration can easily be plugged in our model. We evaluated whether our recommender is able to find a pathway towards the career goal, and whether such a pathway is shorter with respect to the user's one in terms of career hops. The recommended pathway is inherently safer, because it has been obtained by solving an MDP that takes into account all possible career evolutions.

<h2>Run demo</h2>

mdp_cpr_demo.py performs a career pathway recommendation and outputs the following metrics:
- whether the goal state is reached
- whether the recommended pathway is shorter than the original one
- the average length of both pathways

The file can be run as follows: <br/>
``
python mdp_cpr_demo.py
``

Optional parameters are:
- <b>jpf</b> to change the fraction of job postings to be used (default: 0.01)
- <b>nc</b> to change the number of clusters (default: 40)
- <b>spc</b> to choose a different number of skills per cluster (default: 100)
- <b>muh</b> to evaluate the metric only for users having at least <i>muh</i> distinct jobs (default: 10)
- <b>ddir</b> to choose the directory wherein the dataset is contained (default: current directory)
- <b>sdir</b> to choose the directory wherein the scripts are contained (default: current directory)

pathway_inspection_demo.py allows inspecting the user's pathways as well as the recommended one. 
For each user, the output is the following:
- the list of jobs performed
- the actual pathway, once mapped into clusters
- the recommended pathway

The file can be run as follows: <br/>
``
python pathway_inspection_demo.py
``

Optional parameters are:
- <b>uid</b> to compute the pathway for one user only (default: None)
- <b>mu</b> to change the number of users the pathway is computed for (default: 10)
- <b>ddir</b> to choose the directory wherein the dataset is contained (default: current directory)
- <b>sdir</b> to choose the directory wherein the scripts are contained (default: current directory)

The following is an example of an output: <br/>
``
Job sequence of user 50613: ['Line Production Operator', 'Part-Time Package Handler', 'RAN -Transmission Engineer / Site Supervisor, Bharti Airtel Project', 'IB-TI / Site Supervisor, Bharti Airtel Project', 'CTO, Airtel Project and Tigo Milicom Congo Project', 'IB-TI / Site Supervisor, Bahrti Airtel and Tigo Milicom Congo Project', 'Site Inspector, Tigo Milicom Congo Project', 'Driver / Responsible of automotive', 'Data Entry Operator', 'Security Officer']
``
<br/>
``
Pathway of user 50613: [0,25,25,25,0,0,0,15,0]
``
<br/>
``
Recommended pathway to user 50613: [0]
``
