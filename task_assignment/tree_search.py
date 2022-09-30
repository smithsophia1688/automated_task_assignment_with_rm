# holds node class which builds the tree
# also runs the depth first tree search via recursion 

import task_assignment.helper_functions as hf
from reward_machines.sparse_reward_machine import SparseRewardMachine as sparse
import task_assignment.bisimilarity_check as bs
class Node:
    def __init__(self, name = None, children = None, value = -1, knapsack = None, future_events = None, all_events = None, depth = 0):
        if name: ## 'root' -> next_event_name= self.future_events[0]
            self.name = name
        else: 
            self.name = 0

        if children: ## NO
            self.children = children
        else: 
            self.children = [] 

        self.value = value ## -1 in 'root' (not passed) -> 1 or 0 (yes/no in knapsack)

        if knapsack: ## configs.forbidden_events in 'root' -> new_knapsack knapsack.union({next_event_name}) ONLY FOR v =1 
            self.knapsack = knapsack
        else:
            self.knapsack = set()

        if type(future_events) == list: ## configs.future_events in 'root' -> next_events = self.future_events[1:]
            self.future_events = future_events
        else:
            self.future_events = []

        if all_events: ## configs.all_events in 'root'  -> self.all_events
            self.all_events = all_events

        else:
            self.all_events = set()
        self.depth = depth ## 0 in 'root' (not passed) -> new_depth (depth +1)

    def __repr__(self):
        s =  "(" + str(self.name) + ", " + str(self.value) + ")"
        return s 

    def add_children(self, children):
        '''
        adds children to self.children list
        Inputs
            children: (list) holding Node types '''
        self.children.extend(children)
    
    def remove_child(self, child):
        '''
        removes the child from the list of children 
        Inputs
            child: (Node) 
        '''
        if child in self.children:
            self.children.remove(child)
        else: 
            s = str(child) + "was not in" + str(self.name) + "'s children"
            print(s)

    def run_check(self, configs):
        '''
        No forbidden set since tree search already has forbidden set in knapsack and no forbidden events in tree search levels

        '''
        event_spaces, agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, self.knapsack)
        
        # Get projected rm to put in parallel 

        rms = []
        for es in event_spaces:
            rm_p = bs.project_rm(es, configs.rm) 
            rms.append(rm_p)

        rm_parallel = bs.put_many_in_parallel(rms) 

        is_decomp = True
        if self.value != 0: # only check if you changed something
            is_decomp = bs.is_decomposable(configs.rm, rm_parallel, agent_event_spaces_dict, configs.num_agents, enforced_set = configs.enforced_set)

        return is_decomp

    def get_score(self, configs):
        '''
        Get the score of a knapsack according to Fairness and Utility and shared events
        The issue is the current weights are not equal. Need to get a way to have normalized values 
        - shared event score in particluar needs to be normalized some how 
        '''

        event_spaces, event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, self.knapsack)

        se  = configs.get_shared_event_score(self.knapsack)
        f = configs.get_fairness_score(self.knapsack, event_spaces_dict)
        u = configs.get_utility_score(self.knapsack)
        se_weight, f_weight, u_weight = configs.weights
        score = se * se_weight + f * f_weight + u * u_weight
        #print(f"se score is:, {se} , f score is: {f}, u score is: {u}, for a total of {score}")
        
        return score 

    def small_traverse(self, configs, best_sack = (0, [])):
        '''
        Recursive function
        Completes a depth first tree search
        Inputs:
            num_agents: (int)
            rm: (SparseRewardMachine instance)
            config: (Configurations instance)
            best_sack: (int, list of sets) holds the knapsack with the best score and a list of all knapsacks (sets with elements (e,a)) that have that score
            enforced_set: (set) {(e,a)} sets that cannot go into the knapsack
            forbidden_set: (set {(e,a)} set that must go into the knapsack (In configs, we automatically remove the forbidden assignments from the future events so I DO NOT THINK THIS IS NEEDED. )
        '''
        rm = configs.rm
        
        if self.depth <= 4:
            prints = True
        else: 
            prints = False

        if prints:
            ## PRINT STATEMENTS ##
            s_start = '\t |' * self.depth +"--"
            s = s_start + str(self) + " acting on knapsack " + str(self.knapsack) + " with " + str(len(self.future_events)) + " future events"
            print(s)
            #######################

        is_decomp = self.run_check(configs)

        if prints:
            ## PRINT STATEMENTS ##
            s_header = "\t |" * (self.depth) + "\t  "
            if is_decomp:
                s = s_header + "... decomposible"
            else:
                s = s_header + "... NOT decomposible"
            print(s)
            #######################


        # if decomposible and remaining events to add, add children with values 1 & 0
        if is_decomp:
            if self.future_events: 
                # build children 
                next_event_name  = self.future_events[0]
                next_events = self.future_events[1:]
                new_depth = self.depth + 1
                new_knapsack = self.knapsack.union({next_event_name}) # will only be used for child_1

                # add children
                child_1 = Node(name = next_event_name, value = 1, knapsack = new_knapsack, future_events = next_events , all_events = self.all_events, depth = new_depth)
                child_0 = Node(name = next_event_name, value = 0, knapsack = self.knapsack, future_events = next_events , all_events = self.all_events, depth = new_depth)
                self.add_children([child_1, child_0])

            if not self.children:
                if prints:
                    ## PRINT STATEMENTS ## 
                    s = s_header + "SUCCESSFUL TERMINAL, knapsack size = " + str(len(self.knapsack))
                    print(s)
                    #######################

                # Update sucsessful find 
                knap_score = self.get_score(configs)

                if knap_score >= best_sack[0]:

                    if knap_score == best_sack[0]:
                        best_sack[1].append(self.knapsack) # this could get really long 
                    else:
                        best_sack = (knap_score, [self.knapsack])

        else: 
            if not self.children:
                if prints:
                    ## PRINT STATEMENTS ## 
                    s = s_header + "is a FAILED TERMINAL "
                    print(s)
                    #######################
                
        # Recursion Step: 
        for child in self.children:
            best_sack = child.small_traverse(configs, best_sack = best_sack)
        
        if prints:
            ## PRINT STATEMENTS ## 
            s = '\t |'* self.depth +  "  Exiting " +str(self)+ " with best sack = " + str(best_sack[0])
            print(s)
            s = "\t |" * (self.depth-  1) + '\t'
            print(s )
            #######################

        return best_sack


