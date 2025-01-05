import torch

class LSTMModel(torch.nn.Module):

    def __init__(self, config):
        # Initialize super
        super(LSTMModel, self).__init__()

        # The LSTM layer
        self.sequence_network = torch.nn.LSTM(config.input_neuron, config.hidden_neuron, batch_first = True)

        # The FC output Layer
        self.decoder_network_layers = [
            int(config.hidden_neuron),
            int(config.hidden_neuron // 2)
        ]
        self.decoder_network = torch.nn.Sequential(
            torch.nn.Linear(self.decoder_network_layers[0], self.decoder_network_layers[1]),
            torch.nn.Linear(self.decoder_network_layers[1], config.output_neuron)
        )

    def forward(self, x, hidden_cells):
        x, (h_0, c_0) = self.sequence_network(x, hidden_cells)
        # x = self.sequence_network(x, hidden_cells)
        x = self.decoder_network(h_0)
        return x