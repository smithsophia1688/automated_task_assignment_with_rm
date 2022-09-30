class EquivalenceRelation:
    '''
    Relation should be:
        - reflexive: a ~ a
        - symmetric: a ~ b -> b ~ a
        - transitive: a ~ b, b ~ c -> a ~ c
    '''
    def __init__(self, classes = None):
        '''
        Inputs
        classes: list of sets
        
        Attributes
            classes: list holding a set of each equivalence class
        '''
        if classes == None:
            self.classes = []
        else:
            self.classes = classes

        self.check_classes()

    def __repr__(self):
        s = "Equivalence Classes: \n"
        for cl in self.classes:
            s += "    " + str(cl) + "\n"
        return s

        
    def add_relation(self, elements):
        '''
        Inputs
            elements: (list) holds strings of elements to be added to related * MAYBE CHANGE TO TUPLE?
        '''
        #print("     I am adding a relation", elements)
        cls = []
        for e in elements:
            cl = self.find_class(e) 
            if cl:
                if cl not in cls:
                    cls.append(cl)
    
        if cls: # at least one element belongs in a class already
            new_cl = self.merge_classes(cls)  #if only one element in cls, returns cls[0]
            self.add_elements_to_class(elements, new_cl)  
        
        else:
            self.add_new_class(elements) 

        self.check_classes() 

    def find_class(self, element):
        '''
        Finds class the element belongs to
        Input:
            element: (str) (or integer?) element in an equivalence class
        Returns:
            cl: (set) The set containing the element if element is already in a class
                (None) if element is not in a set
        '''
        for cl in self.classes:
            if element in cl:
                return cl
        #print('Element', element, "classes", self.classes)
        return None 
            
    def merge_classes(self, cls):
        '''
        Takes a list of classes, combines them
        Removes existing classes and replaces as ONE 
        Expecting a list of classes at least length 1
        Inputs:
            cls: (list) contains sets that are elements of the list self.classes
        Return:
            megacl: (set) merged classes
        '''
        megacl = set()
        for cl in cls:
            megacl = megacl.union(cl) # build merged class
            self.classes.remove(cl) # remove individual classes

        self.classes.append(megacl) # add merged class
        return megacl
        
    def add_elements_to_class(self, elements, cl):
        '''
        add an element to the set
        Inputs:
            elements: (list) contains strings or int to be added to cl
            cl: (set) equivalence class 
        '''  
        for e in elements:
            cl.add(e)

    def add_new_class(self, elements):
        '''
        Adds a set containting entries in element to list of classes

        elements: List or tuple of elements 
        '''
        new_class = set()
        for e in elements:
            new_class.add(e)
        self.classes.append(new_class)

    def check_classes(self):
        '''
        checks that your equivalence classes are disjoint 
        '''
        full_union = set()
        
        for cl in self.classes:
            for e in cl:
                if e in full_union:
                    raise NameError(" Classes are not disjoint")
                full_union.add(e)

    def are_related(self, elements):

        '''
        Checks if elements belong to the same equivalence class
        Inputs:
            elements: (list or tuple) 
        Returns:
            Bool, true if elements are related, false otherwise 
        '''  
        cls = []
        for e in elements:
            e_cl = self.find_class(e)
            if e_cl == None: # element is not in any equivalence class, elements cannot be related
                return False 
            cls.append(e_cl)
        if len(cls) == 1:
            return True

        if len(cls) == 0: 
            Warning("You are asking if an empty set of elements are related. Returned False")
        
        return False #elements belonged to more than one equivalenc class

    def get_all_related_combos(self):
        '''
        Get all pairs of related elements. Order does not matter.

        For example, if my classes are [{1,2,3}, {4,5}]
        all related combos would be {{1,2}, {1,3}, {2,3}, {4,5}}
        
        Returns: list set of sets
        '''
        all_related_combos = set()
        for cl in self.classes:
            related_combos= set(itertools.combinations(cl, 2))
            for x in related_combos:
                all_related_combos.add(x)
        return all_related_combos    
