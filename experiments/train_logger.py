import json
import time
import os
import numpy as np



# This ResultsLogger class implementation is adapted from the following source:
# https://github.com/Felix-Petersen/difflogic/blob/main/experiments/results_json.py
class ResultsLogger(object):

    def __init__(self, path: str):
        self.path = path
        self.init_time = time.time()
        self.save_time = None
        self.total_time = None
        self.args = None

    def store_args(self, args):
        self.args = vars(args)

    def store_results(self, results: dict):
        for key, val in results.items():
            if not hasattr(self, key):
                setattr(self, key, list())
            getattr(self, key).append(val)

    def store_final_results(self, results: dict):
        for key, val in results.items():
            key = key + '_'
            setattr(self, key, val)

    def save(self):
        self.save_time = time.time()
        self.total_time = self.save_time - self.init_time
        json_str = json.dumps(self.__dict__, cls=NumpyEncoder, indent=4)
        with open(os.path.join(self.path, 'results.json'), mode='w') as f:
            f.write(json_str)

    @staticmethod
    def load(path: str, get_dict=False):
        with open(os.path.join(path, 'results.json'), mode='r') as f:
            data = json.loads(f.read())
        if get_dict:
            return data

        self = ResultsLogger('')
        self.__dict__.update(data)
        return self


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)



class WandbLogger:
    def __init__(self, enabled=True, project_name='project'):
        self.enabled = enabled
        if self.enabled:
            global wandb, torch, plt
            import wandb, torch, os, logging
            import matplotlib.pyplot as plt
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            logging.getLogger('PIL').setLevel(logging.WARNING)
            os.environ["WANDB_SILENT"] = "true"

            wandb.init(project=project_name)
            self.data_accumulator = {}

    def log_metrics(self, metrics, step=None):
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_params(self, params):
        if self.enabled:
            wandb.config.update(params)

    def log_vector(self, data, name, step=None):  # helper function
        if name not in self.data_accumulator:
            self.data_accumulator[name] = []
        self.data_accumulator[name].append(np.copy(data))

    def log_model_weights(self, model, step=None):
        def _get_tau():
            try:
                idx = int(name.split('.')[1])
                if 'neuron_weights' in name:
                    return model.layers[idx].neuron_tau
                elif 'link_weights' in name:
                    return model.layers[idx].link_tau
                elif 'sum_weights' in name:
                    return model.layers[idx].tau
            except AttributeError:
                return 1

        if self.enabled:
            for name, param in model.named_parameters():
                if any(substring in name for substring in ['neuron_weights', 'link_weights']):
                    tau = _get_tau()
                    w_softmax = torch.nn.functional.softmax(param / tau, dim=-1)
                    w_softmax = w_softmax.detach().cpu().numpy()
                    self.log_vector(w_softmax.max(axis=-1), f"{name}.softmax_max", step=step)
                    if 'neuron_weights' in name:
                        self.log_vector(w_softmax.argmax(axis=-1), f"{name}.argmax", step=step)
                if any(substring in name for substring in ['sum_weights']):
                    tau = _get_tau()
                    w_sigmoid_max = torch.sigmoid(param / tau).detach().cpu().numpy().max(axis=-1)
                    self.log_vector(w_sigmoid_max, f"{name}.sigmoid_max", step=step)
                elif any(substring in name for substring in ['th_bias', 'th_slope']):
                    self.log_vector(param.detach().cpu().numpy(), name, step=step)

    @staticmethod  # helper function
    def get_activation(name, accumulators):
        def hook(model, inp, out):
            accumulators[name]['activations'].append(out.detach())
        return hook

    @staticmethod  # helper function
    def get_gradient(name, accumulators):
        def hook(model, grad_input, grad_output):
            accumulators[name]['gradients'].append(grad_output[0].detach().abs())
        return hook

    def log_layer_acts_and_grads(self, model, data_loader, criterion, step=None):
        if self.enabled:
            accumulators = {}
            for name, module in model.named_modules():
                module_name = type(module).__name__
                if module_name not in ['ThresholdLayer', 'LogicLayer', 'SumLayer']:
                    continue
                accumulators[name] = {'activations': [], 'gradients': []}
                activation_hook = self.get_activation(name, accumulators)
                module.register_forward_hook(activation_hook)
                if any(param.requires_grad for param in module.parameters()):
                    gradient_hook = self.get_gradient(name, accumulators)
                    module.register_full_backward_hook(gradient_hook)

            device = next(model.parameters()).device
            for inp, target in data_loader:
                model.zero_grad()
                inp, target = inp.to(device), target.to(device)
                output = model(inp)
                if output.requires_grad:
                    loss = criterion(output, target)
                    loss.backward()

            for layer_name, data in accumulators.items():
                activations_mean = torch.mean(torch.cat(data['activations'], dim=0), dim=0).cpu().numpy()
                self.log_vector(activations_mean, f"{layer_name}.output", step=step)
                data['activations'].clear()
                if data['gradients']:
                    gradients_mean = torch.mean(torch.cat(data['gradients'], dim=0), dim=0).cpu().numpy()
                else:
                    gradients_mean = np.zeros(activations_mean.shape)
                self.log_vector(gradients_mean, f"{layer_name}.gradient_abs", step=step)
                data['gradients'].clear()

    def log_final_data(self):
        for name, data in self.data_accumulator.items():
            data_2d = np.vstack(data).T
            self.log_heatmap(data_2d, name)
        self.data_accumulator.clear()

    def log_heatmap(self, data, name):
        plt.figure(figsize=(10, 5))
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.grid(False)
        wandb.log({f"{name}.heatmap": wandb.Image(plt)})
        plt.close()

    def finish(self):
        if self.enabled:
            self.log_final_data()
            wandb.finish()


