import sys
import os

## importa classes
from environment import Env

from agentes.explorador.explorer import Explorer
from agentes.resgate.rescuer import Rescuer

def main(data_folder_name):
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    # Instantiate the environment
    env = Env(data_folder)

    # config files for the agents
    explorer_file = os.path.join(data_folder, "explorer_config.txt")
    rescuer_file = os.path.join(data_folder, "rescuer_config.txt")

    # Instantiate agents rescuer and explorer
    resc = Rescuer(env, rescuer_file)

    # Explorer needs to know rescuer to send the map
    # that's why rescuer is instatiated before
    exp = Explorer(env, explorer_file, resc)

    # Run the environment simulator
    env.run()

data_folder_name = "./data_folder/"

if __name__ == '__main__':
    # To get data from a different folder than the default called data pass it by the argument line

    if len(sys.argv) > 1:
        data_folder_name += sys.argv[1]
    else:
        data_folder_name += "data"
        # data_folder_name += "data_treino1"
        # data_folder_name += "TESTE 2"
        # data_folder_name += "TESTE 3"

    main(data_folder_name)
