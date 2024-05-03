# Environment troubleshooting

Here are some issues I came across during setting up the Python environment.
Ray only works with Python 3.7 so you should set up the environment like:

` conda create -n forl python=3.7 `
` conda activate forl `

Some prerequisites to avoid version incompatibility, (don't install anything prior to this, otherwise you can start over):

- To be able to use that specific Ray version that they provided, you also need the following
```  
    pip install wheel==0.38.0 setuptools==65.5.0 pip==21
    pip install protobuf==3.20.*
```
Now you can install the rest:

```
    pip install -e .
    pip install rl-warp-drive --no-dependencies
    pip install gym==0.21
    pip install tensorflow==1.14
    pip install "ray[rllib]==0.8.4"
```

For Ray you need to uninstall this...:

```
    pip uninstall pyarrow
```

Now you can import everything and it should work.

If you have issues installing `pycuda` when installing `rl-warp-drive` you need to issue `sudo apt install nvidia-cuda-toolkit` (on linux for GPU)