from preprocess import read_data, split_data, full_process
from models import motor
import os
import argparse
from pathlib import Path
import yaml
import pickle
import tensorflow as tf
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
from telegram import send_telegram

BASE_DIR = Path(__file__).resolve().parent

neptune_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NWE2M2I5My1iZjFmLTRhOWItOGEyNy01YjBlYzMwZmQzNWIifQ=="


def train(config_path):

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    data_url = config['data_url']
    batch_size = config['batch_size']
    shuffle = config['shuffle']

    train, valid = read_data(data_url)

    train_generator, valid_generator = full_process(train, valid, batch_size, shuffle)

    n_neurons_lstm = config['n_neurons_lstm']
    n_neurons_timedistributed = config['n_neurons_timedistributed']
    learning_rate = config['learning_rate']

    model = motor(
       n_neurons_lstm,
       n_neurons_timedistributed,
       learning_rate
    )

    epochs = config['epochs']
    type_device = config['type_device']
    type_model = config['type_model']

    #Monitor
    run = neptune.init_run(
    name = type_model,  
    project="ammar.mlops/arabic-loneha",
    api_token=neptune_token)
    url_project = run.get_url()
    
    send_telegram("The URL ML Track for model: "
                  + f"<b>{type_model}</b> 🤓"
                  + "\nPlease Use <b> VPN </b>😅 \n"
                  + f"{url_project}\n.")

    neptune_callback = NeptuneCallback(run=run,
                                       log_model_diagram=True)
    if type_device == "GPU":
      with tf.device(f'/{type_device}:0'):
        history = model.fit(
           train_generator,
           validation_data=valid_generator,
           epochs=epochs,
           callbacks=[neptune_callback],
        )
    
    result_path = os.path.join(BASE_DIR, os.path.join("outputs", f"model_{type_model}.h5"))
    model.save(result_path)


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Process some URLs.')
   parser.add_argument('--config_path', type=str, help='The URL for configuration train')
   args = parser.parse_args()

   config_path = args.config_path
   train(config_path)



