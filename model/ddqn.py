import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

class DoubleDeepQNetworkTagetModel():
    def __init__(self):
        self.model = Sequential()
        # [WriteCode]
        # model.add(...

        # Compile the model
        # [WriteCode]

        # Print the model summary
        # [WriteCode]
        
    def predict(self, state, verbose=0):
        '''
        给定环境状态 输出 qvalues
        '''
        pass
    
    def compile(self):
        '''
        不应该实现 应该保持和 eval_model 一致
        '''
        pass
    
    
    def set_weights(self, weights):
        '''
        设置模型权重 应该保持和 eval_model 一致
        '''
        pass
    
class DoubleDeepQNetworkEvalModel():
    def __init__(self):
        self.model = Sequential()
        # [WriteCode]
        # model.add(...

        # Compile the model
        # [WriteCode]

        # Print the model summary
        # [WriteCode]
        
    def predict(self, state, verbose=0):
        '''
        给定环境状态 输出 qvalues
        '''
        pass
    
    def compile(self):
        pass
    
    def get_weights(self):
        '''
        获取模型权重
        '''
        pass