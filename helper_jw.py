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
    time = (end-start) / 60
    est_m = time * (bn-b)
    print(f"\nTime taken for batch: {int(time)} min(s)")
    print(f"Estimated remaining time: "
          f"{int(est_m / 60)}:{int(est_m % 60) if int(est_m % 60) > 10 else f'0{int(est_m % 60)}'} hrs:min")
print("Finished all!")
