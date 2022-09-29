
### generic functions to move between knapsack and agent:event_spaces dictionaries ###

def get_event_spaces_from_knapsack(all_events, knapsack = None): 
    '''
    I feel like this is not very good code in this.
    takes all (event, agent) pairs, removes those in knapsack
    and converts it to {agent: [event]}
    and also [{events for agent 1}, {events for agent 2}, ... ]

    '''
    if not knapsack:
        knapsack = set()

    event_spaces_dict = {}
    remaining_events = all_events - knapsack 

    for e, a in remaining_events:
        if a not in event_spaces_dict.keys():
            event_spaces_dict[a] = [e]
        else:
            event_spaces_dict[a].append(e)
    
    event_spaces = []
    for agent, event_list in event_spaces_dict.items():
        event_spaces.append(set(event_list))

    return event_spaces, event_spaces_dict
    
def get_sack_from_dict(dict):
    '''
    takes a dict with form {a: [e1, e2]}
    and returns set: {(a,e1), (a,e2)}
    '''
    knapsack = set()
    for a, es in dict.items():
        for e in es:
            knapsack.add((e,a))
    return knapsack
        

