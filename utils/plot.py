import matplotlib.pyplot as plt

def plot_loss(epoch, train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch, train_loss, label='train loss', marker = 'o', linestyle = '-',color='blue')
    plt.plot(epoch, val_loss, label='val loss', marker = 'o', linestyle = '--',color='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def plot_acc(epoch, val_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch, val_acc, label='val acc', marker = 'o', linestyle = '--',color='red')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()

