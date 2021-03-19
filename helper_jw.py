from tqdm import tqdm
from chalearn_jw import process_images

# Current batch, batch size, total number of batches, left and right "borders" for batch
b = 331
bs = 740
bn = 2252
l = (b-1) * bs
r = b * bs
for i in tqdm(range(bn-b+1)):
    process_images(b, bn, l, r)
    b += 1
    l += bs
    r += bs
print("\n\nAll done!!!")


# Leftover code -> save for later maybe
'''
plasma = plt.get_cmap('plasma')
a = item[:, :, 0]
a -= np.min(a)
a /= np.max(a)
a = plasma(a)[:, :, :3]
'''
