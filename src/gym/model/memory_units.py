
class Memory_unit:
    def __init__(self, DQN, _id, min_experiences, max_experiences):
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.parent = DQN
        self.id = _id
        self.current_observation = None
        self.reset()

    @staticmethod
    def empty_observation():
        return {
            's': None,
            'a': None,
            'r': None,
            's2': None,
            'done': None
        }

    @staticmethod
    def key_translation(key):
        translation = {
            'prev_state': 's',
            'action': 'a',
            'reward': 'r',
            'state': 's2',
            'done': 'done'
        }
        return translation[key]

    def assign_property(self, key, value): # return true if the property is not None
        if value == None:
            return self.current_observation[key] != None
        self.current_observation[key] = value
        return True

    def reset(self):
        self.current_observation = self.empty_observation()
        return self

    def observe(self, prev_state=None, action=None, reward=None, state=None, done=None):
        args = locals()
        done = True
        for key, arg in args.items():
            if not self.assign_property(
                self.key_translation(key),
                arg
            ):
                done = False
        
        if done:
            self.add_experience(self.current_observation)
            self.reset()
    
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)
    

    
    
    