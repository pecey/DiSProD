
class PID:

    def __init__(  self,dt,  Kp,  Kd,  Ki ):
        self._dt = dt
        self._Kp = Kp
        self._Kd = Kd
        self._Ki = Ki
        self._pre_error = 0
        

    
    def calculate( self,error,  max_ ):
        Pout = self._Kp * error


        derivative = (error - self._pre_error) / self._dt
        Dout = self._Kd * derivative
        output = Pout + Dout


        if (output > max_):
        
            output = max_
        

        elif(output < 0) and (output < -max_):
        
            output = -max_
        

        self._pre_error = error

        return output
