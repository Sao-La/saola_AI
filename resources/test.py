names = set()
with open("./vi-animals.txt", "r") as f:
    for x in f:
        names.add(x.strip())

names = sorted(names)
with open("./vi-animals.txt", "w") as f:
    for x in names:
        f.write(x + "\n")
