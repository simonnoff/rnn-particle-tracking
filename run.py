import math
import matplotlib.pyplot as plt
from scripts.data_processor import DataLoader
from scripts.lstm_network import Model

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def main():
    data = DataLoader(
        "data/generated_points_normalised.csv",
        0.9,
        ["x", "y"]
    )

    sequence_length = 50
    batch_size = 30
    normalise = True
    epochs = 2

    model = Model()
    model.build_model()
    x, y = data.get_train_data(
        seq_len=sequence_length,
        normalise=normalise
    )

    steps_per_epoch = math.ceil((data.len_train - sequence_length) / batch_size)
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=sequence_length,
            batch_size=batch_size,
            normalise=normalise
        ),
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        save_dir="scripts/saved_checkpoint"
    )

    x_test, y_test = data.get_test_data(
        seq_len=sequence_length,
        normalise=normalise
    )

    #predictions = model.predict_sequences_multiple(x_test, sequence_length, sequence_length)
    predictions = model.predict_sequence_full(x_test, sequence_length)
    #predictions = model.predict_point_by_point(x_test)

    #plot_results_multiple(predictions, y_test, sequence_length)
    plot_results(predictions, y_test)

if __name__ == '__main__':
    main()

