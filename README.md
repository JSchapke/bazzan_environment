## Multiagent route choice environment

Contains code for:
Inducing selfish agents towards social efficient solutions *by João Schapke and Ana Bazzan* (https://doi.org/10.5753/kdmile.2020.11953).

Visualization of the environment:

![Image of the environment](https://github.com/JSchapke/routing-environment/blob/master/visualization.png)

Implemented solvers:
- QLearning
- Genetic Algorithm
- GA-QL

### Setup
Run:
```
pip install -r requirements.txt
```

### QLearning
To run qlearning execute:
```
python q_learning.py <netfile> 
```

### Genetic Algorithm
To run ga execute:
```
python ga.py <netfile> 
```

###  GA - QL
To run GA-QL execute:
```
python ga_ql.py <netfile> 
```

### Tuning the models
To check all the tunable parameters for an algorithm use:
```
python <script>.py --help
```
