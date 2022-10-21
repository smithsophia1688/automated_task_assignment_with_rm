# holds node class which builds the tree
# also runs the depth first tree search via recursion 

import task_assignment.helper_functions as hf
from reward_machines.sparse_reward_machine import SparseRewardMachine as sparse
import task_assignment.bisimilarity_check as bs



class Node:
    def __init__(self, name = None, children = None, value = -1, knapsack = None, future_events = None, all_events = None, depth = 0, doomed = True):
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
        
        self.doomed = doomed

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
        #if self.value != 0: # only check if you changed something
        is_decomp = bs.is_decomposable(configs.rm, rm_parallel, agent_event_spaces_dict, configs.num_agents, enforced_set = configs.enforced_set, incompatible_pairs = configs.incompatible_pairs, upcomming_events = self.future_events)

        return is_decomp

    def run_check_2(self, configs):

        if configs.type != 'no_accidents':
            print("Freak out not ready for accidents yet")

        event_spaces, agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, self.knapsack)

        restrictions_pass, have_hope = bs.check_restrictions(configs, agent_event_spaces_dict, self.future_events)

        if restrictions_pass: # here, we satisfied all of our restrictions
            bisim_check, check_children = bs.is_decomposible_no_accidents(configs, event_spaces) 

        else:
            if have_hope: 
                bisim_check = False
                check_children = True # default is to check all children 
                # restrictions have failed but there is hope they are fixed. What should I do? 

            else:
                bisim_check = False
                check_children = False

        return bisim_check, check_children #need to incorperate this into my tree search. 
            
    def check_win_ability(self, configs, event_spaces):
        '''
        This function checks if removing the assingments 
        in a given knapsack (leaving the event_spaces) leaves
        a winnable strategy
        '''
        strategy_set = set()
        for es in event_spaces:
            strategy_set = strategy_set.union(es)

        strategic_rm = bs.remove_rm_transitions(configs.rm, strategy_set) 
        bs.remove_unreachable_states(strategic_rm)

        if not bs.can_win_check(strategic_rm):
            self.doomed = False 
        
    def check_sufficent_agents(self, configs, agent_event_spaces_dict):
        active_agents = []
        for a, es in agent_event_spaces_dict.items():
            if es:
                active_agents.append(a)
        if len(active_agents) != configs.num_agents:
            # you have removed too many agents
            self.doomed = False


    def check_incompatible_assignments(self, configs, agent_event_spaces_dict):
        pass

    def run_check_last_minute(self, configs): 
        event_spaces, agent_event_spaces_dict = hf.get_event_spaces_from_knapsack(configs.all_events, self.knapsack)
        
        if not self.doomed:
            self.check_sufficent_agents(configs, agent_event_spaces_dict)
        if not self.doomed:
            self.check_incompatible_assignments()
        if not self.doomed:
            self.check_win_ability(configs, event_spaces)
        

        
    # need a function to check if I have a winning strategy (return True/ False) DONE
    # need a function to check if I have incompatible assignments and if I have hope for kids 
    # need a function that checks if a valid node is bisimilar 
    # need a function that coordinates this and records through node flags. (i hate flags)

    # flags I will need: node_doomed, value (1 or 0) did I change something, is_valid (True if "happy", False if "sad")
    # Not I can be is_valid = False and not be doomed (depends on how I got there and where I am going)
    
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
            s_start = '\t |' * self.depth + "--"
            s = s_start + str(self) + " with " + str(len(self.future_events)) + " future events"
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
        
        if is_decomp: # I think I need to get rid of this condition? or is it I should not fail the decomp.. 

            if self.future_events: 
                # build children 

                next_event_name  = self.future_events[0]
                next_events = self.future_events[1:]
                new_depth = self.depth + 1
                new_knapsack = self.knapsack.union({next_event_name}) # will only be used for child_1

                # add children
                child_1 = Node(name = next_event_name, value = 1, knapsack = new_knapsack, future_events = next_events, all_events = self.all_events, depth = new_depth)
                child_0 = Node(name = next_event_name, value = 0, knapsack = self.knapsack, future_events = next_events, all_events = self.all_events, depth = new_depth)
                self.add_children([child_1, child_0])

            if not self.children:
                if prints:
                    ## PRINT STATEMENTS ## 
                    s = s_header + "SUCCESSFUL TERMINAL, knapsack size = " + str(len(self.knapsack))
                    print(s)
                    #######################

                # Update sucsessful find 
                knap_score = configs.get_score(self.knapsack)

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

    def new_traverse(self, configs, best_sack = (0, [])):

        if self.depth <= 9:
            prints = True
        else: 
            prints = False

        if prints:
            ## PRINT STATEMENTS ##
            s_start = '\t |' * self.depth + "--"
            s = s_start + str(self) + " with " + str(len(self.future_events)) + " future events"
            print(s)
            #######################

        # step 1: check if our knapsack is decomposible and if we should check the kids
        is_decomp, check_kids = self.run_check_2(configs)
        #craft_check = False
        #if ('craft', 2) in self.knapsack: 
        #    craft_check = True
        #    print(self.knapsack, is_decomp, check_kids)

        # step 2: if check kids, add kids. 
        if check_kids: 
            if self.future_events:  # Make children if I can
                next_event_name  = self.future_events[0]
                next_events = self.future_events[1:]
                new_depth = self.depth + 1
                new_knapsack = self.knapsack.union({next_event_name}) # will only be used for child_1

                # Add children
                child_1 = Node(name = next_event_name, value = 1, knapsack = new_knapsack, future_events = next_events, all_events = self.all_events, depth = new_depth)
                child_0 = Node(name = next_event_name, value = 0, knapsack = self.knapsack, future_events = next_events, all_events = self.all_events, depth = new_depth)
                self.add_children([child_1, child_0])

            else: # there are no future event events to add as kids
                if is_decomp: # If decomposible, I should collect data on this node. 
                    knap_score = configs.get_score(self.knapsack)
                    if knap_score >= best_sack[0]:
                        if knap_score == best_sack[0]:
                            best_sack[1].append(self.knapsack) # this could get really long 
                        else:
                            best_sack = (knap_score, [self.knapsack])

        else:  # don't bother checking kids. 
            if is_decomp: # I don't think this can ever happen, if it does, collect score
                print("WTF ")
                knap_score = configs.get_score(self.knapsack)
                if knap_score >= best_sack[0]:
                    if knap_score == best_sack[0]:
                        best_sack[1].append(self.knapsack) # this could get really long 
                    else:
                        best_sack = (knap_score, [self.knapsack])

        
        # Recursion Step: 
        for child in self.children:
            best_sack = child.new_traverse(configs, best_sack = best_sack)

        return best_sack

