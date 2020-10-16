import numpy as np

def show_progress(i, loops, s="training"):
    if(i%int(loops/100)==0): print("{} progress: {}%".format(s, int(i/(loops/100))))

def activation_function(x):
    def sigmoid(element):
        return 1/(1+np.exp(-element))
    return np.vectorize(sigmoid)(x)

class new_network:    
    def __init__(self, dim_input, dim_reservoir, dim_output, var_of_weight_recurrent = 1.5, weight={}, readout="zero"):
        self.dim_input = dim_input
        self.dim_reservoir = dim_reservoir
        self.dim_output = dim_output
        var = var_of_weight_recurrent
        if(weight =={}):
            weight_recurrent = np.random.normal(0, np.sqrt(var**2/dim_reservoir), [dim_reservoir, dim_reservoir])
            # feedback
            weight_feedback = (np.random.randn(dim_reservoir, dim_output))
            # readout
            if readout == "zero":
            	weight_readout = np.zeros([dim_output, dim_reservoir])
            elif readout == "evenly":
            	weight_readout = np.random.rand(dim_output, dim_reservoir)
            elif readout == "normal":
            	weight_readout = np.random.randn(dim_output, dim_reservoir)
            # input weights
            weight_input = 2.0*(np.random.randn(dim_reservoir, dim_input))
            self.weight = {"input": weight_input, 
                           "readout": weight_readout,
                           "recurrent": weight_recurrent,
                           "feedback": weight_feedback}
        else:
            self.weight = weight
        eigenvalues = np.linalg.eigvals(weight_recurrent)
        self.spectral_radius = np.max(np.vectorize(np.linalg.norm)(eigenvalues))
    def run(self, input_for_one_interval, reservoir_state):
        x = reservoir_state
        w_inp = self.weight["input"]
        w_out = self.weight["readout"]
        w_rec = self.weight["recurrent"]
        w_feed = self.weight["feedback"]
        #run echo state network
        value_inp = input_for_one_interval.reshape([self.dim_input, 1])
        value_res = activation_function(x)
        value_out = w_out.dot(value_res)
        x_delta = w_rec.dot(value_res) + w_inp.dot(value_inp) + w_feed.dot(value_out) 
        x = 0.2*x + 0.8*x_delta
        return x, value_res, value_out
    
    def train(self, inp, target, training_speed = 1, train_delta = 2, show_progress_or_not = False):
        w_inp = self.weight["input"]
        w_out = self.weight["readout"]
        w_rec = self.weight["recurrent"]
        w_feed = self.weight["feedback"]
        input_duration = inp.shape[0]
        #initial state
        alpha = 1
        reservoir_state = 0.5*np.random.randn(self.dim_reservoir, 1)
        P = (1.0/alpha)*np.eye(self.dim_reservoir)
        #time constant
        for t in range(input_duration):
            if(show_progress_or_not == True): 
                show_progress(t, input_duration)
            else:
                pass
            x, value_res, value_out = self.run(inp[t], reservoir_state)
            reservoir_state = x        
            if(t%train_delta == 0):
                k = P.dot(value_res)
                rPr = value_res.T.dot(k)
                c = 1/(1 + rPr)
                P = P - k.dot(k.T)*c
                error = value_out - np.expand_dims(target[t], 1)
                w_out_delta = -error*c*k.T
                w_out += w_out_delta*training_speed
        self.weight["readout"] = w_out
        return None
    
    def clear_weight_readout(self):
        self.weight["readout"] = np.zeros([self.dim_output, self.dim_reservoir])
        return None
    
    def test(self, inp, give_me_reservoir_state = False, show_progress_or_not = False):
        duration = inp.shape[0]
        reservoir_state = np.zeros([self.dim_reservoir, 1])
        output = np.empty([duration, self.dim_output])
        reservoir_state_record = np.empty([duration, self.dim_reservoir])
        for t in range(duration):
            if(show_progress_or_not == True): 
                show_progress(t, duration, "testing")
            else:
                pass
            reservoir_state, value_res, value_out = self.run(inp[t], reservoir_state)
            output[t] = value_out[:, 0]
            reservoir_state_record[t] = reservoir_state[:, 0]
        if(give_me_reservoir_state == False):
            return output
        else:
            return output, reservoir_state_record