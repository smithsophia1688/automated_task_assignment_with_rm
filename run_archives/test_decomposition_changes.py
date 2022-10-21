from cmath import exp
from datetime import datetime
import pickle
import os

import reward_machines.sparse_reward_machine as sparse
from task_assignment.configurations import Configurations
import task_assignment.get_decompositions as gd
import task_assignment.bisimilarity_check as bs
import task_assignment.helper_functions as hf
from task_assignment.tree_search import Node 

      

num_agents = 3
rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/testing_decomp_1.txt') 
#rm = sparse.SparseRewardMachine(file = 'data/saved_reward_machines/crafting_task/crafting_rm_full_updated.txt' )
print(rm.events)

'''

print(" Step 1: check that you have a valid original rm")
print("    ... ", bs.can_win_check(rm))

{'tr3', 'a3', 'l1', 'l2', 'cutting', 'l3', 'a1', 'timber', 'ar', 'a2', 'tr1', 'craft', 'tr2', 'st_cutting'}

forbidden_agent_event_dict = {0:['a2', 'l2', 'a3', 'l3', 'craft', 'tr2', 'tr3'], 1:['a1', 'l1', 'a3', 'l3', 'craft', 'tr1', 'tr3'], 2:['a1', 'l1', 'a2', 'l2', 'tr1', 'tr2']} 

es1= [ 'l1', 'cutting', 'a1', 'timber', 'ar', 'tr1',  'st_cutting']
es2 = ['l2', 'cutting', 'timber', 'ar', 'a2',  'tr2', 'st_cutting']
es3 = ['tr3', 'a3',  'cutting', 'l3', 'timber', 'ar', 'st_cutting']

es_list = [set(es1), set(es2), set(es3)]
strategy_set = set()
for es in es_list:
    strategy_set = strategy_set.union(es)

strategic_rm = bs.remove_rm_transitions(rm, strategy_set)
bs.remove_unreachable_states(strategic_rm)
print(strategic_rm)
print(strategic_rm.delta_r)
print("Can I win", bs.can_win_check(strategic_rm))

p1 = bs.project_rm(set(es1), strategic_rm)
p2 = bs.project_rm(set(es2), strategic_rm)
p3 = bs.project_rm(set(es3), strategic_rm )
rms = [p1, p2, p3]

rm_p = bs.put_many_in_parallel(rms)


decomposable = bs.is_bisimilar( strategic_rm, rm_p)

print("decomposable is", decomposable)



weights = [1, 1, 0]
#forbidden_agent_event_dict = {0:['a2', 'l2', 'a3', 'l3'], 1:['a1', 'l1', 'a3', 'l3'], 2:['a1', 'l1', 'a2', 'l2']} 
enforced_agent_event_dict = {0: [], 1:[], 2: []}
incompatible_pairs = []

# DO I need to ask for the "best" strategy? 
# Will that be the default if I pressurize shared events etc. 



configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, weights = weights, incompatible_pairs= incompatible_pairs)



# Initialize tree 
root = Node(name = 'root', future_events = configs.future_events, all_events= configs.all_events, knapsack = configs.forbidden_set) #forbidden set is the starting knapsack

print("tree events", configs.future_events)

# Execute Tree search 
bd = root.new_traverse(configs)
hf.print_results(configs, bd)

#print("BD", bd)
#knapsack = bd[1][0] # arbitrarily taking first knapsack with best score. 
# Unexpected behavior: It is allowing me to get rid of "leaving" as I do not need it. That is probably a valid" input. 



#rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/crafting_task/crafting_rm_full.txt') 


'''

rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/testing_decomp_1.txt') 

es1 = ['a1',  'l1', 'r']
es2 = ['a2', 'l2', 'r']
es3 = []

es_list = [set(es1), set(es2), set(es3)]
strategy_set = set()
for es in es_list:
    strategy_set = strategy_set.union(es)

acc_set = rm.events.copy() - strategy_set
print("accidents;", acc_set) 
#strategy_set = {'a1', 'l1', 'r', 'a2', 'l2'}
strategic_rm = bs.get_strategy_rm(rm, strategy_set)


p1 = bs.project_rm(set(es1), strategic_rm)
p2 = bs.project_rm(set(es2), strategic_rm)
p3 = bs.project_rm(set(es3), strategic_rm )

print("p1 before", p1)

#aap1 = bs.get_accident_avoidance_rm(p1, acc_set)
#aap2 = bs.get_accident_avoidance_rm(p2, acc_set)
#aap3 = bs.get_accident_avoidance_rm(p3, acc_set)

rms = [p1, p2, p3]

print("p1 after", p1)
#print("aap1", aap1)

#aap1.delta_u.clear()
#print(aap1.delta_u)
#print(p1.delta_u)
#print("aap2", aap2)
#print("aap3", aap3)
#print("p2:", p2)
#print("p3:", p3)

rm_p = bs.put_many_in_parallel(rms)

#print("RM P")
#print(rm_p)

decomposable = bs.is_bisimilar( strategic_rm, rm_p)

print("decomposable is", decomposable)
