# Setting the environment on Euler

I have carefully selected all the correct versions for the libraries - at some points overriding the required dependency contrainst - to make it work on Euler.
You have two options:

## Building the environment yourself
I think this is the inferior option, as you have to build some of the libs from source.
Load the required Euler modules: `module load gcc/6.3.0 python/3.6.6 eth_proxy`.
Upgrade pip: `pip install --upgrade pip`.
Create an environment with `python -m venv env --system-site-packages`, activate the environment with `source env/bin/activate` and install the dependencies with `pip install -r requirements.txt`
Install the AI Economist package by `cd ../../..` and `pip install -e .`.

## Using a prebuilt environment
I compressed the built python environment and uploaded it here (~310MB): https://polybox.ethz.ch/index.php/s/Ue4OswsjHljaijK
Download and uncompress the archive to the 'euler' folder with the command `tar -xvjf env.tar.bz2`. Afterwards, you can modify the path to the config in the 'train_euler.sh' and submit it via `sbatch train_euler.sh`.
Since the env was built on Euler the binaries should be compatible. Make sure the env directory stays in the euler folder and it is not being moved otherwise, the relative path dependencies will break.

## Submitting a job
After you have the correct environment sumbit the job via `sbatch train_euler.sh`, the logs will be in the euler/logs folder.