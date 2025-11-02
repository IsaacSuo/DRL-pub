import yaml

class TrainingConfig():
    def __init__(self,
                 lr=0, 
                 epoch=1,
                 episode=250,
                 epsilon=1,
                 epsilon_min=0.01,
                 epsilon_decay=0,
                 gamma=0.99,
                 ba=1,
                 target_update_freq=0,):
        '''
        Use the following set of NN hyperparameters for ALL FOUR baseline policies
        '''
        self.lr =  lr        #@param {type:"number"}               # learning rate
        self.epoch =  epoch     #@param {type:"number"}               # epochs
        self.episode = episode  #@param {type:"number"}               # episodes

        self.epsilon = epsilon           #@param {type:"number"}     # Starting exploration rate
        self.epsilon_min = epsilon_min    #@param {type:"number"}     # Exploration rate min
        self.epsilon_decay = epsilon_decay     #@param {type:"number"}     # Exploration rate decay

        self.gamma = gamma          #@param {type:"number"}     # Agent discount factor

        # Use the following set of NN hyperparameters for Naive DQN, DQN and DDQN policies
        self.ba =  ba       #@param {type:"number"}               # batch_size

        # Use the following set of RL hyperparameters for DQN and DDQN policies
        self.target_update_freq = target_update_freq # @param {type:"number"}    # Target network update frequency
    
    def __repr__(self):
        return f"TrainingConfig(lr={self.lr}, epoch={self.epoch}, episode={self.episode}, epsilon={self.epsilon}, epsilon_min={self.epsilon_min}, epsilon_decay={self.epsilon_decay}, gamma={self.gamma}, ba={self.ba}, target_update_freq={self.target_update_freq})"
    
    def load(self, path):
        with open(path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            self.lr = float(params['lr']) 
            self.epoch =  int(params['epoch'])
            self.episode = int(params['episode'])
            self.epsilon = float(params['epsilon'])
            self.epsilon_min = float(params['epsilon_min'])
            self.epsilon_decay = float(params['epsilon_decay'])
            self.gamma = float(params['gamma'])
            self.ba =  int(params['ba'])
            self.target_update_freq = int(params['target_update_freq'])
        f.close()