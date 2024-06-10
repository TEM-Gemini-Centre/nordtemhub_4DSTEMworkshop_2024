import numpy as np
import torch
import time
import matplotlib.pyplot as plt




def plot_graphs(training, validation=None, ylim=None):
    fig, ax = plt.subplots(1,1, figsize=(8,6), constrained_layout=True)
    ax.plot(training, lw=3, label='Training')
    if validation is not None:
        ax.plot(validation, lw=3, label='Validation')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=16)





def plot_patterns(indices, dataset, gts=None, preds=None):
    n = len(indices)
    assert 1 < n < 5, 'only 1 < n < 5 work'
    fig, axes = plt.subplots(1,n, figsize=((8-n)*n, 8-n), constrained_layout=True)
    for i in range(n):
        axes[i].imshow(dataset.inav[indices[i]], cmap='gray'); axes[i].set_axis_off()
        if gts is not None:
            cent = np.around(gts.inav[indices[i]].data, 2)
            axes[i].scatter(gts.inav[indices[i]].data[1], gts.inav[indices[i]].data[0], c='r', marker='X', s=70, label='ground truth')
            axes[i].text(x=10, y=20, s='Cx, Cy: [%s, %s]'%(cent[0], cent[1]), c='w', fontsize=14, fontweight='bold')
        if preds is not None:
            axes[i].scatter(preds.inav[indices[i]].data[1], preds.inav[indices[i]].data[0], c='g', marker='X', s=70, label='prediction')
            axes[0].legend()





class DatasetLoader(torch.utils.data.Dataset):
    def __init__(self, dataset, gt=None):
        super().__init__()
        self.data = dataset.data
        if gt is not None:
            self.gt = gt.data
        else:
            self.gt = None
                        
    def __len__(self):
        return len(self.data)
     
    def open_array(self, idx):
        dp = np.expand_dims(self.data[idx], axis=0)
        dp = dp/np.max(dp)
        return dp
    
    def __getitem__(self, idx):
        if self.gt is None:
            x = torch.tensor(self.open_array(idx), dtype=torch.float32)
            return x
        else:
            x = torch.tensor(self.open_array(idx), dtype=torch.float32)
            y = torch.tensor(self.gt[idx], dtype=torch.float32)
            return x, y





def train(model, optimizer, loss_function, metric, batch_size, train_data, valid_data, device, num_epochs=10, early_stop=False):
    start_time = time.time()
    train_loss, valid_loss = [], []
    train_met, valid_met = [], []

    for epoch in range(num_epochs):
        print('-' * 100)
        print('Epoch %i/%i'%(epoch, num_epochs - 1))

        for phase in ['Training', 'Validiation']:
            if phase == 'Training':
                model.train(True)
                datal = train_data
            else:
                model.eval()
                datal = valid_data

            running_loss = 0.0
            running_met = 0.0

            step = 0

            for x, y in datal:
                x = x.to(device)
                y = y.to(device)
                step += 1

                if phase == 'Training':
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = loss_function(outputs, y)
                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_function(outputs, y)

                with torch.no_grad(): met = metric(outputs, y)
                running_loss += loss * datal.batch_size
                running_met += met * datal.batch_size
                
                if step % (batch_size) == 0:
                    print('%s; image number: %i, loss: %.4f, metric: %.4f'%(phase, step*datal.batch_size, loss, met))

            epoch_loss = (running_loss / len(datal.dataset)).detach().numpy()
            epoch_met = (running_met / len(datal.dataset)).detach().numpy()
            
            train_loss.append(epoch_loss) if phase=='Training' else valid_loss.append(epoch_loss)
            train_met.append(epoch_met) if phase=='Training' else valid_met.append(epoch_met)

            print('%s epoch loss: %.4f, epoch metric: %.4f'%(phase, epoch_loss, epoch_met)) # print epoch loss + metric

        if early_stop:
            patience = 3
            if (len(valid_loss) > patience) and (len(valid_loss) - np.argmin(np.array(valid_loss)) >= patience):
                print("Early stop at epoch: %s and patience: %s"%(epoch, patience))
                break

    time_elapsed = time.time() - start_time
    print('Training complete, time elapsed: %.0fm %.0fs'%(time_elapsed // 60, time_elapsed % 60))
    
    return np.array(train_loss), np.array(valid_loss), np.array(train_met), np.array(valid_met)