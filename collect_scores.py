from cmath import exp
from datetime import datetime
import pickle
import os
import numpy as np
import reward_machines.sparse_reward_machine as sparse
from task_assignment.configurations import Configurations
import task_assignment.get_decompositions as gd
import task_assignment.bisimilarity_check as bs
import task_assignment.helper_functions as hf
from task_assignment.tree_search import Node 

### PROBABLY DO NOT NEED THE NEXT TWO.. ###
import  Environments.crafting.crafting_env as env
import Environments.crafting.multiagent_crafting_env as multi_env
import time
import matplotlib.pyplot as plt
start_time = time.time()

#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#
######################################           SET UP             #######################################
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#

# experiment_name = 'crafting'
experiment_name = 'five'

weights = [1, .5, 0]

random = False
trivial = False
best = True
centralized = False
decomposition_types = [random, trivial, best, centralized]

if experiment_name == 'crafting':
    num_agents = 3
    #rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/crafting_task/crafting_rm_full_no_cut.txt') # I would really like this to work haha
    rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/crafting_task/crafting_rm_full_no_cut.txt') # I would really like this to work haha
    print(rm.events)
    forbidden_agent_event_dict = {0:['a2', 'l2', 'a3', 'l3', 'craft', 'tr2', 'tr3'], 1:['a1', 'l1', 'a3', 'l3', 'craft', 'tr1', 'tr3'], 2:['a1', 'l1', 'a2', 'l2', 'tr1', 'tr2']} 
    agent_utility_function = {0:{'timber': 0, 'tr2': 0, 'ar':0, 'a1': 3, 'craft':0, 'l3':0, 'a3':0, 'tr3':0, 'a2':0, 'l1':0, 'tr1':0, 'l2':0}, \
        1: {'timber':0, 'tr2':0, 'ar':0, 'a1':0, 'craft':0, 'l3':0, 'a3':0, 'tr3':0, 'a2': 2, 'l1':0, 'tr1':0, 'l2':0}, \
        2:{'timber':0, 'tr2':0, 'ar':0, 'a1':0, 'craft':0, 'l3':0, 'a3':1, 'tr3':0, 'a2':0, 'l1':0, 'tr1':0, 'l2':0}}

    enforced_agent_event_dict = {0:[], 1:[], 2:[]}
    #forbidden_agent_event_dict = {0:['craft'], 1:['craft'], 2:[]}
    incompatible_pairs = []
    configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, weights = weights, incompatible_pairs= incompatible_pairs,agent_utility_function = agent_utility_function )

elif experiment_name == 'five':
    num_agents = 5
    rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/five_agent/five_agent_team_rm_early_consequences.txt') # I would really like this to work haha
    print(rm.events)
    forbidden_agent_event_dict = {0:['e2', 'e3', 'e4', 'e5'], 1:['e1', 'e3', 'e4', 'e5'], 2:['e2', 'e1', 'e4', 'e5'], 3: ['e1', 'e2', 'e3', 'e5'], 4: ['e1', 'e2', 'e3', 'e4']} 
    #enforced_agent_event_dict = {0: ['a1', 'l1', 'timber'], 1:['a2', 'l2', 'timber', 'tr2', 'ar'], 2: ['ar', 'craft']}
    agent_utility_function = {0:{'e1':1, 'e4':0, 'e5':0, 'b':0, 'a':0, 'e2':0, 'e3':0}, \
        1: {'e1':0, 'e4':0, 'e5':0, 'b':0, 'a':0, 'e2':4, 'e3':0}, \
        2:{'e1':0, 'e4':0, 'e5':0, 'b':0, 'a':0, 'e2':0, 'e3':2},\
            3: {'e1':0, 'e4':5, 'e5':0, 'b':0, 'a':0, 'e2':0, 'e3':0}, \
                4: {'e1':0, 'e4':0, 'e5':3, 'b':0, 'a':0, 'e2':0, 'e3':0}}

    enforced_agent_event_dict = {0:[], 1:[], 2:[]}
    incompatible_pairs = []
    include_all = False
    configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, weights = weights, incompatible_pairs= incompatible_pairs, include_all = include_all, agent_utility_function = agent_utility_function)


#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#
######################################       TASK ASSIGNMENT        #######################################
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#


root = Node(name = 'root', future_events = configs.future_events, all_events= configs.all_events, knapsack = configs.forbidden_set) #forbidden set is the starting knapsack
print("tree events", configs.future_events)
print(len(configs.future_events))
print("ways: ", 2**len(configs.future_events))

#bd = root.new_traverse(configs)

bd = root.traverse_collect_all_valid(configs)
# hf.print_results(configs, bd)
#knapsack = bd[1][0] # arbitrary pick 

print("--- %s seconds ---" % (time.time() - start_time))
#knapsack_id = input(" Which TA do you want? ")
# bd = {score: [] , score: [] }

print(len(bd))
scores_0 = []

set_weights = [1, 0, 0]
for ks in bd:
    score = configs.get_score(ks, set_weights)
    scores_0.append(score)
    if score == 1.0:
        print("ks is", ks)
        print("kept sets:")
        es_list , agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, ks)
        print(es_list)
        print("  ")
print("max score", max(scores_0))

scores_1 = []
set_weights = [0, 1, 0]
for ks in bd:
    score = configs.get_score(ks, set_weights)
    scores_1.append(score)
    

scores_2 = []
set_weights = [0, 0, 1]
for ks in bd:
    score = configs.get_score(ks, set_weights)
    scores_2.append(score)

scores_3 = []
set_weights = [.333, .333, .333]
for ks in bd:
    score = configs.get_score(ks, set_weights)
    scores_3.append(score)

# bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
bins = np.linspace(0, 1, 15)
plt.hist(scores_0, bins = bins,  label = '[1, 0, 0]', color = 'blue', alpha = .5) #lw=3, fc=(1, 0, 0, 0.5)) 
plt.hist(scores_1, bins = bins, label = '[0, 1, 0]', color = 'green', alpha = .5) #lw=3, fc=(0, 1, 0, 0.5)) 
plt.hist(scores_2, bins = bins, label =  '[0, 0, 1]',color= 'red', alpha = .5) #lw=3, fc=(0, 0, 1, 0.5)) 
plt.hist(scores_3, bins = bins, label = '[.333, .333, .333]',color = 'k', alpha = .5) #lw=3, fc=(.33, .33, .33, 0.5)) 
plt.legend()
plt.title(f"Distribution of Scores With Different Weights, {experiment_name}")
plt.xlabel("Scores")
plt.ylabel("Number of Decompositions")
plt.show()


print("max score [1,0,0]", max(scores_0))
print("max score [0,1,0]", max(scores_1))
print("max score [0,0,1]", max(scores_2))
print("max score [.33,.33,.33]", max(scores_3))
# bins = []
# counts = []
# for s, ks in bd.items():
#     bins.append(s)
#     counts.append(len(ks))

# plt.plot(bins, counts, '.r')

#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#
######################################   PRE-LEARNING PROCESSING    #######################################
#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#+#


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
# sample usage

# Okay , so i have the plot of different scores
# Still need a way to pull out extremes 
# Also still need a way to test out on learning. Lets do learning comparisons. Be cool. Make a effing difference 
# Do the 2 extreme cases. 






