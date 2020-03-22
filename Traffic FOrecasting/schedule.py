import os
all_files = os.listdir('input')
for files in all_files:
    print('starting...')
    os.system('python extract.py --video input/'+ files)
print('finished...')