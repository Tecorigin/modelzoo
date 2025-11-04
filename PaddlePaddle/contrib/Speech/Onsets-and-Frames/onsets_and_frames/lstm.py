import paddle
from paddle import nn


class BiLSTM(nn.Layer):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features,direction = 'bidirectional'):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, direction = direction, name = 'onsets_and_frames_biLSTM')
        self.bidirectional = True if direction == 'bidirectional' else False

    def forward(self, x):
        if self.training:
            x = self.rnn(x)
            x = x[0]
            return x
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.bidirectional else 1

            h = paddle.zeros([num_directions, batch_size, hidden_size,])
            c = paddle.zeros([num_directions, batch_size, hidden_size,])
            output = paddle.zeros([batch_size, sequence_length, num_directions * hidden_size])

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.bidirectional:
                h = paddle.zeros_like(h)
                c = paddle.zeros_like(c)

                for start in (slices[::-1]):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output
