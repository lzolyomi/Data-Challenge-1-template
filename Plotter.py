import matplotlib.pyplot as plt

# Plotting some images
unique_labels = set(class_labels.keys())
fig, ax = plt.subplots(ncols=len(unique_labels), figsize=[25, 5])

for k, label in enumerate(unique_labels):
    ind = list(train_y).index(label)
    # ax[k].imshow(train_x[ind].reshape(128,128), cmap='gray')
    ax[k].set_title(f"Class:{class_labels[train_y[ind]]}")

fig.savefig("test.png")
