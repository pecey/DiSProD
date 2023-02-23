To install the dependencies, `pip install requirements.txt`

The launcher code is located in main.py. The default config for environments are defined in this file.  

The solver for each environment is present in the folders for the environment. These files contain the next_step and reward methods for each of the environment.
 
Cart Pole : python main.py --environment=cart_pole
Mountain Car : python main.py --environment=mountain_car
Continuous Dubins Car : python main.py --environment=continuous_dubins_car
Continuous Mountain Car : python main.py --environment=continuous_mountain_car
