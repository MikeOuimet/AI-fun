class MineHunt(object):
    def __init__(self, **kwargs):
        self.start = 1
        self.h= '1'

    def legal_actions(self, s):
        return [1, 2, 3, 4]


    def rollout_policy(self, s, h):
        '''chooses actions for rollout policy'''
        # draw from self.legal_action(s)
        return self.legal_actions(s)[s]