from tqdm import tqdm
from process_jw import process_images

# Current batch, total number of batches, left and right "borders" for batch
b = 193
bn = 2252
l = (b-1) * 740
r = b * 740
for i in tqdm(range(bn-b+1)):
    process_images(b, bn, l, r)
    b += 1
    l += 740
    r += 740
print("\n\n!!!All done!!!")
