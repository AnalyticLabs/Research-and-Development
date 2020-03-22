import os
files = os.listdir('input')
for f in files:
    src = 'input/'+f
    dest = 'input/'+f.split('2019m09')[1]
    os.rename(src,dest)
print(os.listdir('input'))