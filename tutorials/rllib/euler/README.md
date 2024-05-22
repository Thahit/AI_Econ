# Setting the environment on Euler

I have carefully selected all the correct versions for the libraries - at some points overriding the required dependency contrainst - to make it work on Euler.
You have two options:

## Using a prebuilt environment
I compressed the built python environment and uploaded it here (~300MB): 
Download and uncompress the archive to the 'euler' folder with the command `tar -xvjf env.tar.bz2`. Afterwards, you can modify the path to the config in the 'train_euler.sh' and submit it via `sbatch train_euler.sh`.
Since the env was built on Euler the binaries should be compatible.

## Building the environment yourself
I think this is the inferior option, as you have to build some of the libs from source and the whole process might take 0.5-1h.
Create an environment with `python -m venv env --system-site-packages`, activate the environment with `source env/bin/activate` and install the dependencies with `pip install -r requirements.txt`

