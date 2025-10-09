# cartpole-dqn
Vrije Universiteit Amsterdam - 2025 - Introduction to Reinforcement Learning Group Project (Team 2)

## Environment

The code is based on PyTorch and Gynmasium.

The dependency can be installed into a virtual environment as follows.

For CPU only:

```
pip install -r requirements.txt -r requirements-cpu.txt
```

Or with GPU

```
pip install -r requirements.txt -r requirements-gpu.txt
```

## Train the DQN

From the project root directory, create the network weights

```
python -O src/base.py
```

To run a custom training, use the cli flags. The list of parameters can be
obtained with the help.

```
python src/base.py --help
```

## Run the Test

Run the test script

```
python tests/test_script.py
```

