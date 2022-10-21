from cmath import exp
from datetime import datetime

import reward_machines.sparse_reward_machine as sparse
from task_assignment.configurations import Configurations
import task_assignment.get_decompositions as gd


experiment_name = 'buttons'
#experiment_name = 'rendezvous_small'

num_agents = 3      
#num_agents = 2

weights = [5, 1, 1]

#### Pick your decompositions to test ####
random = False
trivial = True
best = False
centralized = False
decomposition_types = [random, trivial, best, centralized]




if experiment_name == 'buttons' and num_agents == 3:
    rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/buttons/team_buttons_rm.txt') # I would really like this to work haha
    
    # Decomposition we expect (used only for comparison in analysis), mostly for reference and posterity 
    expected_set = {0:['by', 'br', 'g'], 1: ['bg', 'a2br', 'a2lr', 'by', 'br'], 2: ['a3br', 'a3lr', 'br', 'bg']}

    # Set Restrictions 
    enforced_agent_event_dict  = {0:[], 1:[], 2:[]}        # {agent : [required  events]}
    forbidden_agent_event_dict = {0:[], 1:['g'], 2:['g']}  # {agent : [forbidden events]}

    # Set agent utility of each event
    agent_utility_function = {0 : {'by': 0, 'bg': 0 , 'g':1 , 'a2br': 0 , 'a2lr': 0 , 'a3br' : 0, 'a3lr':0, 'br':0},
                              1 : {'by': 0, 'bg': 0 , 'g':0 , 'a2br': 1 , 'a2lr': 1 , 'a3br' : 0, 'a3lr':0, 'br':0},
                              2 : {'by': 0, 'bg': 0 , 'g':0 , 'a2br': 0 , 'a2lr': 0 , 'a3br' : 1, 'a3lr':1, 'br':0}}    # { agent : {event: utility} }

    # Build configurations class
    configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, agent_utility_function = agent_utility_function, weights = weights)
    
    experiment_time = datetime.now().strftime("%Y_%m_%d-%I.%M.%S_%p")
    new_rm_file_location = 'data/saved_reward_machines/buttons/'
    
    if best:
        best_decomposition = gd.get_best_decomposition(configs, new_rm_file_location, experiment_time )

    if trivial:
        trivial_decomposition = gd.get_trivial_decomposition(configs, new_rm_file_location, experiment_time)
    
    if random:
        random_decomposition = gd.get_random_decompositions(configs, new_rm_file_location, experiment_time )


elif experiment_name == 'rendezvous_small' and num_agents == 2:
    print("this far")
    rm =  sparse.SparseRewardMachine(file = 'data/saved_reward_machines/gridworld_many_agent_rendezvous/2_agent_rendezvous_rm_small.txt') # I would really like this to work haha

    print(rm.events)
    # Decomposition we expect (used only for comparison in analysis), mostly for reference and posterity 
    expected_set = {0:['l1', 'r1', 'r'], 1: ['l2', 'r2', 'r']}
    
    # Set Restrictions 
    enforced_agent_event_dict  = {0:['l1'], 1:['l2']}        # {agent : [required  events]}
    forbidden_agent_event_dict = {0:[], 1:[]}                # {agent : [forbidden events]}

    agent_utility_function = {0 : {'l1': 0, 'r1': 0 , 'l2':1 , 'r2': 0 , 'r': 0 },
                              1 : {'l1': 0, 'r1': 0 , 'l2':1 , 'r2': 0 , 'r': 0 }}    # { agent : {event: utility} }

    configs = Configurations(num_agents, rm, enforced_set = enforced_agent_event_dict, forbidden_set = forbidden_agent_event_dict, agent_utility_function = agent_utility_function, weights = weights)
    
    experiment_time = datetime.now().strftime("%Y_%m_%d-%I.%M.%S_%p")
    new_rm_file_location = 'data/saved_reward_machines/rendezvous_small/'

    if best:
        best_decomposition = gd.get_best_decomposition(configs, new_rm_file_location, experiment_time )
        print('best files are', best_decomposition[0])

    if trivial:
        trivial_decomposition = gd.get_trivial_decomposition(configs, new_rm_file_location, experiment_time)
        print('trivial files are', trivial_decomposition[0])
  
    if random:
        random_decomposition = gd.get_random_decomposition(configs, new_rm_file_location, experiment_time )
        print('random files are', random_decomposition[0])

else: 
    print(f'The {experiment_name} experiment with {num_agents} agents is not defined yet!') # this is a silly print statement 
