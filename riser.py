import time
from timeit import default_timer as timer

from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import numpy as np
from read_until import ReadUntilClient
from read_until.read_cache import AccumulatingCache
import torch
from torchinfo import summary

from cnn import ConvNet
from utilities import get_config
from preprocess import SignalProcessor


# TODO: Logging
# TODO: Multithreading
# TODO: Annotate function signatures (arg types, return type)
# TODO: Comments


def classify(signal, device, model):
    with torch.no_grad():
        X = torch.from_numpy(signal).unsqueeze(0)
        X = X.to(device, dtype=torch.float)
        logits = model(X)
        return torch.argmax(logits, dim=1)


def analysis(client, model, device, processor, target, duration=0.1, throttle=0.4, batch_size=512):    
    while client.is_running:
        # Pass through current batch of reads retrieved from client
        start_t = time.time()
        unblock_batch_reads = []
        stop_receiving_reads = []
        for (channel, read) in client.get_read_chunks(batch_size=batch_size,
                                                      last=True):
            # Preprocess raw signal if it's long enough
            signal = np.frombuffer(read.raw_data, client.signal_dtype)
            if len(signal) < processor.get_required_length():
                continue
            signal = processor.process(signal)

            # Accept or reject read
            prediction = classify(signal, device, model)
            if prediction != target:
                unblock_batch_reads.append((channel, read.number))

            # Don't need to assess the same read twice
            stop_receiving_reads.append((channel, read.number))

        # Send reject requests
        if len(unblock_batch_reads) > 0:
            client.unblock_read_batch(unblock_batch_reads, duration=duration)
        if len(stop_receiving_reads) > 0:
            client.stop_receiving_batch(stop_receiving_reads)

        # Limit request rate
        end_t = time.time()
        if start_t + throttle > end_t:
            time.sleep(throttle + start_t - end_t)
    else:
        print("Client stopped, finished analysis.")


def setup_client():
    client = ReadUntilClient(filter_strands=True,
                             one_chunk=False,
                             cache_type=AccumulatingCache)

    # Start communication with MinKNOW
    client.run(first_channel=1, last_channel=512)

    # Make sure client is running
    while client.is_running is False:
        time.sleep(0.1)
    
    return client


def setup_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")
    return device


def setup_model(model_file, config_file, device):
    config = get_config(config_file)
    model = ConvNet(config.cnn).to(device)
    model.load_state_dict(torch.load(model_file))
    summary(model)
    model.eval()
    return model


def main():
    # CL args
    config_file = './local_data/configs/train-cnn-20.yaml'
    model_file = 'local_data/models/train-cnn-20_0_best_model.pth'
    polyA_length = 6481
    input_length = 12048
    target = 'protein-coding'
    

    # Set up
    client = setup_client()
    device = setup_device()
    model = setup_model(model_file, config_file, device)
    processor = SignalProcessor(polyA_length, input_length)
    target_class = 1 if target == 'protein-coding' else 0 # TODO: primitive obsession

    # Run analysis
    # TODO: Is ThreadPoolExecutor needed? Readfish just calls analysis
    # function directly.
    # with ThreadPoolExecutor() as executor:
    #     executor.submit(analysis, read_until_client)
    analysis(client, model, device, processor)

    client.reset()

if __name__ == "__main__":
    main()

