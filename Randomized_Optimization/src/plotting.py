import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getNNPlot(path, input, output, title, x_lable, y_label, y_index1, y_index2, x_range):
    result = pd.read_csv('./../' + path + '/' + input + '.txt')
    x = range(0, x_range, 10)
    y1 = result[y_index1].tolist()
    y2 = result[y_index2].tolist()
    plt.figure()
    plt.plot(x, y1, label='train')
    plt.plot(x, y2, label='validation')
    plt.title(title)
    plt.xlabel(x_lable)
    plt.ylabel(y_label)
    plt.legend()
    
    plt.savefig('./../img/' + output + '.png')

def getSAPlot(input, title, y_label):
    result1 = pd.read_csv('./../NN_OUTPUT/SA_' + input + '_0.15_LOG.txt')
    result2 = pd.read_csv('./../NN_OUTPUT/SA_' + input + '_0.35_LOG.txt')
    result3 = pd.read_csv('./../NN_OUTPUT/SA_' + input + '_0.55_LOG.txt')
    result4 = pd.read_csv('./../NN_OUTPUT/SA_' + input + '_0.7_LOG.txt')
    result5 = pd.read_csv('./../NN_OUTPUT/SA_' + input + '_0.95_LOG.txt')
    x = range(0, 5001, 50)
    y1 = result1[y_label].tolist()
    y2 = result2[y_label].tolist()
    y3 = result3[y_label].tolist()
    y4 = result4[y_label].tolist()
    y5 = result5[y_label].tolist()
    plt.figure()
    plt.plot(x, y1, label='CE-0.15')
    plt.plot(x, y2, label='CE-0.35')
    plt.plot(x, y3, label='CE-0.55')
    plt.plot(x, y4, label='CE-0.7')
    plt.plot(x, y5, label='CE-0.95')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('./../img/SA_' + input + '.png')


def getGAPlot(input, title, y_label):
    result1 = pd.read_csv('./../NN_OUTPUT/GA_' + input + '_10_10_LOG.txt')
    result2 = pd.read_csv('./../NN_OUTPUT/GA_' + input + '_10_20_LOG.txt')
    result3 = pd.read_csv('./../NN_OUTPUT/GA_' + input + '_20_10_LOG.txt')
    result4 = pd.read_csv('./../NN_OUTPUT/GA_' + input + '_20_20_LOG.txt')
    x = range(0, 5001, 50)
    y1 = result1[y_label].tolist()
    y2 = result2[y_label].tolist()
    y3 = result3[y_label].tolist()
    y4 = result4[y_label].tolist()
    plt.figure()
    plt.plot(x, y1, label='10-10')
    plt.plot(x, y2, label='10-20')
    plt.plot(x, y3, label='20-10')
    plt.plot(x, y4, label='20-20')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('./../img/GA_' + input + '.png')


def getPart2SAPlot(input, title, y_label, iterations):
    result1 = pd.read_csv('./../' + input + '/' + input + '_SA0.15_1_LOG.txt')
    result2 = pd.read_csv('./../' + input + '/' + input + '_SA0.35_1_LOG.txt')
    result3 = pd.read_csv('./../' + input + '/' + input + '_SA0.55_1_LOG.txt')
    result4 = pd.read_csv('./../' + input + '/' + input + '_SA0.75_1_LOG.txt')
    result5 = pd.read_csv('./../' + input + '/' + input + '_SA0.95_1_LOG.txt')
    x = range(0, iterations, 10)
    y1 = result1[y_label].tolist()
    y2 = result2[y_label].tolist()
    y3 = result3[y_label].tolist()
    y4 = result4[y_label].tolist()
    y5 = result5[y_label].tolist()
    plt.figure()
    plt.plot(x, y1, label='SA-0.15')
    plt.plot(x, y2, label='SA-0.35')
    plt.plot(x, y3, label='SA-0.55')
    plt.plot(x, y4, label='SA-0.75')
    plt.plot(x, y4, label='SA-0.95')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Evaluation Values')
    plt.legend()

    plt.savefig('./../img/' + input + '_SA.png')


def getPart2GAPlot(input, title, y_label, iterations):
    result1 = pd.read_csv('./../' + input + '/' + input + '_GA100_10_10_1_LOG.txt')
    result2 = pd.read_csv('./../' + input + '/' + input + '_GA100_10_30_1_LOG.txt')
    result3 = pd.read_csv('./../' + input + '/' + input + '_GA100_30_10_1_LOG.txt')
    result4 = pd.read_csv('./../' + input + '/' + input + '_GA100_30_30_1_LOG.txt')
    x = range(0, iterations, 10)
    y1 = result1[y_label].tolist()
    y2 = result2[y_label].tolist()
    y3 = result3[y_label].tolist()
    y4 = result4[y_label].tolist()
    plt.figure()
    plt.plot(x, y1, label='GA-10-10')
    plt.plot(x, y2, label='GA-10-30')
    plt.plot(x, y3, label='GA-30-10')
    plt.plot(x, y4, label='GA-30-30')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Evaluation Values')
    plt.legend()

    plt.savefig('./../img/' + input + '_GA.png')


def getPart2MIMICPlot(input, title, y_label, iterations):
    result1 = pd.read_csv('./../' + input + '/' + input + '_MIMIC100_50_0.1_1_LOG.txt')
    result2 = pd.read_csv('./../' + input + '/' + input + '_MIMIC100_50_0.3_1_LOG.txt')
    result3 = pd.read_csv('./../' + input + '/' + input + '_MIMIC100_50_0.5_1_LOG.txt')
    result4 = pd.read_csv('./../' + input + '/' + input + '_MIMIC100_50_0.7_1_LOG.txt')
    result5 = pd.read_csv('./../' + input + '/' + input + '_MIMIC100_50_0.9_1_LOG.txt')
    x = range(0, iterations, 10)
    y1 = result1[y_label].tolist()
    y2 = result2[y_label].tolist()
    y3 = result3[y_label].tolist()
    y4 = result4[y_label].tolist()
    y5 = result4[y_label].tolist()
    plt.figure()
    plt.plot(x, y1, label='MIMIC-100-50-0.1')
    plt.plot(x, y2, label='MIMIC-100-50-0.3')
    plt.plot(x, y3, label='MIMIC-100-50-0.5')
    plt.plot(x, y4, label='MIMIC-100-50-0.7')
    plt.plot(x, y4, label='MIMIC-100-50-0.9')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Evaluation Values')
    plt.legend()

    plt.savefig('./../img/' + input + '_MIMIC.png')


def getPart2RHCPlot(input, title, y_label, iterations):
    result = pd.read_csv('./../' + input + '/' + input + '_MIMIC100_50_0.1_1_LOG.txt')
    x = range(0, iterations, 10)
    y = result[y_label].tolist()
    plt.figure()
    plt.plot(x, y, label='RHC')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Evaluation Values')
    plt.legend()

    plt.savefig('./../img/' + input + '_RHC.png')


def main():
    getNNPlot('NN_OUTPUT', 'BP_LOG', 'BP_accuracy', 'BP Accuracy', 'Iterations', 'Accuracy', 'acc_trg', 'acc_val', 5001)
    getNNPlot('NN_OUTPUT', 'BP_LOG', 'BP_MSE', 'BP MSE', 'Iterations', 'MSE', 'MSE_trg', 'MSE_val', 5001)
    getNNPlot('NN_OUTPUT', 'RHC_LOG2', 'RHC_accuracy', 'RHC Accuracy', 'Iterations', 'Accuracy', 'acc_trg', 'acc_val', 3001)
    getNNPlot('NN_OUTPUT', 'RHC_LOG2', 'RHC_MSE', 'RHC MSE', 'Iterations', 'MSE', 'MSE_trg', 'MSE_val', 3001)
    getSAPlot('1000000.0', 'SA Accuray - Temperature of 1E6', 'acc_val')
    getSAPlot('100000000.0', 'SA Accuray - Temperature of 1E8', 'acc_val')
    getSAPlot('10000000000.0', 'SA Accuray - Temperature of 1E10', 'acc_val')
    getSAPlot('1e+12', 'SA Accuray - Temperature of 1E12', 'acc_val')
    getGAPlot('45', 'GA Accuray - Population of 45', 'acc_val')
    getGAPlot('55', 'GA Accuray - Population of 55', 'acc_val')
    getPart2SAPlot('TSP', 'TSP SA Evaluation Curves', 'fitness', 3001)
    getPart2SAPlot('CONTPEAKS', 'CONTPEAKS SA Evaluation Curves', 'fitness', 5001)
    getPart2SAPlot('FLIPFLOP', 'FLIPFLOP SA Evaluation Curves', 'fitness', 3001)
    getPart2GAPlot('TSP', 'TSP GA Evaluation Curves', 'fitness', 3001)
    getPart2GAPlot('CONTPEAKS', 'CONTPEAKS GA Evaluation Curves', 'fitness', 5001)
    getPart2GAPlot('FLIPFLOP', 'FLIPFLOP GA Evaluation Curves', 'fitness', 3001)
    getPart2MIMICPlot('TSP', 'TSP MIMIC Evaluation Curves', 'fitness', 3001)
    getPart2MIMICPlot('CONTPEAKS', 'CONTPEAKS MIMIC Evaluation Curves', 'fitness', 5001)
    getPart2MIMICPlot('FLIPFLOP', 'FLIPFLOP MIMIC Evaluation Curves', 'fitness', 3001)
    getPart2RHCPlot('TSP', 'TSP RHC Evaluation Curves', 'fitness', 3001)
    getPart2RHCPlot('CONTPEAKS', 'CONTPEAKS RHC Evaluation Curves', 'fitness', 5001)
    getPart2RHCPlot('FLIPFLOP', 'FLIPFLOP RHC Evaluation Curves', 'fitness', 3001)


if __name__ == "__main__":
    main()