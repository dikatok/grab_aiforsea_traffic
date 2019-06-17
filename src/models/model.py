import tensorflow as tf
import time
import os


class Encoder(tf.keras.Model):
    def __init__(self, units=128):
        super(Encoder, self).__init__()
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, enc_in, hidden):
        enc_in = tf.cast(enc_in, dtype=tf.float32)
        output, state = self.gru(enc_in, initial_state=hidden)
        return output, state


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, units=32):
        super(Decoder, self).__init__()
        self.gru = tf.keras.layers.GRU(units,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(50)
        self.fc2 = tf.keras.layers.Dense(1)
        self.attention = BahdanauAttention(units)

    def call(self, enc_out, hidden):
        context_vector, attention_weights = self.attention(hidden, enc_out)
        context_vector = tf.tile(tf.expand_dims(context_vector, 1), multiples=[1, enc_out.shape[1], 1])
        dec_in = tf.concat([context_vector, enc_out], axis=-1)
        output, state = self.gru(dec_in)
        output = self.fc1(output)
        output = self.fc2(output)
        return output, state, attention_weights


class Model:
    def __init__(self):
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(20))
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint_dir = "../models"
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)

    @tf.function
    def train_step(self, inputs, outputs, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            inputs = self.dense(inputs)

            enc_output, enc_hidden = self.encoder(inputs, enc_hidden)

            predictions, dec_hidden, _ = self.decoder(enc_output, enc_hidden)

            loss += tf.reduce_mean(tf.losses.mean_squared_error(outputs, predictions))

        batch_loss = (loss / int(outputs.shape[0]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train(self, dataset):
        start = time.time()

        enc_hidden = None
        total_loss = 0
        iteration = 0

        summary_writer = tf.summary.create_file_writer("../logs")

        for (batch, (inputs, outputs)) in enumerate(dataset):
            batch_loss = self.train_step(inputs, outputs, enc_hidden)
            total_loss += batch_loss

            if batch % 10 == 0:
                print('Iter {} Batch {} Loss {}'.format(iteration + 1,
                                                            batch,
                                                            batch_loss.numpy()))

                with summary_writer.as_default():
                    tf.summary.scalar('loss', batch_loss, step=iteration)

        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        print('Iter {} Loss {}'.format(iteration + 1, total_loss / iteration))
        print('Time taken {} sec\n'.format(time.time() - start))
