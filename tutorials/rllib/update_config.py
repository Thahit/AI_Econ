import os
import yaml

def find_weights_filename(directory):
    for filename in os.listdir(directory):
        #print(filename)
        if filename.startswith("agent.tf.weights.global-step-"):
            print(filename)
            return filename
    return None

def update_configure_yaml(filename):
    with open("/home/bahain/forl/runs/phase2/config.yaml", "r") as file:
        config = yaml.safe_load(file)


    config["general"]["restore_tf_weights_agents"] = '/home/bahain/forl/runs/phase1/ckpts/' + filename

    print(config)
    print(filename)

    with open("/home/bahain/forl/runs/phase2/config.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False)

if __name__ == "__main__":
    directory = '/home/bahain/forl/runs/phase1/ckpts/'
    filename = find_weights_filename(directory)
    if filename:
        update_configure_yaml(filename)
        print("Filename added to configure.yaml successfully.")
    else:
        print("No matching filename found in the directory.")
