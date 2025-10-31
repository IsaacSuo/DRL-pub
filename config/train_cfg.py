class TrainingConfig():
    def __init__(self,
                 lr, 
                 epoch,
                 episode,
                 epsilon,
                 epsilon_min,
                 epsilon_decay,
                 gamma,
                 ba,
                 target_update_freq,):
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