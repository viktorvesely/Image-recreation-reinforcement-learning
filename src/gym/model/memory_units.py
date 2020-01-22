
class Memory_unit:
    def __init__(self, DQN, id, min_experiences, max_experiences):
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.parent = DQN
        self.id = id

    def observe(self, prev_state, action, reward, state, done):
        self.add_experience({
            's': prev_state,
            'a': action,
            'r': reward,
            's2': state,
            'done': done
            })
    
    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)
    

    
    
    