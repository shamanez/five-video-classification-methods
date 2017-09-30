"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt

def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies = []
        loss = []
        cnn_benchmark = []  # this is ridiculous
        for epoch,acc,loss,val_acc,val_loss in reader:
            
            accuracies.append(float(val_acc))
            #cnn_benchmark.append(0.65)  # ridiculous

        plt.plot(accuracies)
        plt.plot(loss)
        #plt.plot(cnn_benchmark)
        plt.show()

if __name__ == '__main__':
    training_log = 'data/logs/lrcn-training-1506694471.13.log'
    main(training_log)
