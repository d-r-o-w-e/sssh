# run ALL scripts starting with genfig

import os, sys, subprocess

regen_npys = sys.argv[1]
quick = sys.argv[2]
startat = sys.argv[3] if len(sys.argv) >= 4 else None

if not (startat is None):
    print("Attempting to start at" + startat)

started = False

files_run = []

for filename in sorted(os.listdir(".")):
    print(filename)
    if not (startat is None) and not started and filename != startat:
        continue
    started = 1
    if filename[:6] == "genfig":
        print("generating using " + filename)
        files_run += [filename]
        os.system("python3 " + filename + " " + str(regen_npys) + " " + str(quick))

print("ALL THE FILES THAT WERE RUN:")
print(files_run)