import timeit
from test_jw import process_images


b = 1
bn = 2252
l = 0
r = 740
for i in range(bn):
    start = timeit.default_timer()
    process_images(b, bn, l, r)
    b += 1
    l += 740
    r += 740
    end = timeit.default_timer()
    time = (start-end) // 60
    est_m = time * (bn-b)
    print(f"\nTime taken for batch: {time} mins")
    print(f"\nEstimated remaining time: {est_m // 60} hours, {est_m % 60} mins")
print("Finished all!")
