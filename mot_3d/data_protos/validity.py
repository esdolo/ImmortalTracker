class Validity:
    TYPES = ['birth', 'alive', 'death']
    def __init__(self):
        return
    
    @classmethod
    def valid(cls, state_string):
        tokens = state_string.split('_')
        if not tokens[0] == 'alive':
            return False
        if tokens[0] == 'alive' and int(tokens[1]) == 1:
            return True
        return False

    @classmethod
    def agein_n(cls, state_string,n):
        tokens = state_string.split('_')
        if tokens[0] in ['birth','death']:
            return False
        try:
            if int(tokens[2]) <= n-1:
                return True
        except:
            import pdb
            pdb.set_trace()
        return False

    @classmethod
    def agein2(cls, state_string):
        tokens = state_string.split('_')
        if tokens[0] == 'dead':
            return False
        try:
            if int(tokens[2]) <= 1:
                return True
        except:
            import pdb
            pdb.set_trace()
        return False
        
    @classmethod
    def agein1(cls, state_string):
        tokens = state_string.split('_')
        if tokens[0] == 'dead':
            return False
        try:
            if int(tokens[2]) <= 0:
                return True
        except:
            import pdb
            pdb.set_trace()
        return False
    
    @classmethod
    def notoutput(cls, state_string):
        tokens = state_string.split('_')
        if len(tokens) < 3:
            return False
        if tokens[0] == 'alive' and int(tokens[1]) != 1:
            return True
        return False
    
    @classmethod
    def predicted(cls, state_string):
        state, token = state_string.split('_')
        if state not in Validity.TYPES:
            raise ValueError('type name not existed')
        
        if state == 'alive' and int(token) != 0:
            return True
        return False
    
    @classmethod
    def modify_string(cls, state_string, mode):
        tokens = state_string.split('_')
        tokens[1] = str(mode)
        return '{:}_{:}_{:}'.format(tokens[0], tokens[1], tokens[2])