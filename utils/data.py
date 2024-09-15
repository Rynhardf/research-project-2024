# read ./frames.csv
f = open('./frames.csv', 'r')

f.readline()

num_frames = 0
for line in f:
    num_frames += 1

print(f"Number of frames: {num_frames}")