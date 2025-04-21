# AI algorithms design and development for the Mancala game

## Simulations
The AI algorithm is developed inside the ai folder, the simulations are run through 
``` {bash}
python simulations.py
```

## Mancala UI
Run the server by: 
```
python server.py
```

Run the UI frontend by: 
```
python -m http.server 8000
```
And access: localhost:8000/mancala.html

## DQN model training
To train dqn model:
```
python ai/dqn.py
```

Then, to access training results log:
```
python -m tensorboard.main --logdir=ai/runs
```