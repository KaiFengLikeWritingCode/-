import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses, out_path, title):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses,   label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.title(title); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
