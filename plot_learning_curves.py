import matplotlib.pyplot as plt
import csv

train_loss = []
with open('train_loss_xPaiNN_v4_2602855_mda_def2-svp_fixed.txt') as fp:
    contents = csv.reader(fp, delimiter=',')
    for entries in contents:
        for entry in entries[:-1]:
            train_loss.append(float(entry)*27.2114/0.0434)
val_loss = []
with open('val_loss_xPaiNN_v4_2602855_mda_def2-svp_fixed.txt') as fp:
    contents = csv.reader(fp, delimiter=',')
    for entries in contents:
        for entry in entries[:-1]:
            val_loss.append(float(entry)*27.2114/0.0434)

plt.plot(range(len(train_loss)),train_loss,label="train")
plt.plot(range(len(val_loss)),val_loss,label="test")
plt.xlabel("epochs")
plt.ylabel("loss in kcal/mol")
plt.yscale('log')
plt.ylim([0,max(train_loss)])
plt.legend()
plt.show()