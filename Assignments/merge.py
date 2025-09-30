import nbformat

# Read both notebooks
with open('Assignments/Assignment1.1.ipynb', 'r', encoding='utf-8') as f:
    nb1 = nbformat.read(f, as_version=4)

with open('Assignments/Assignment1.ipynb', 'r', encoding='utf-8') as f:
    nb2 = nbformat.read(f, as_version=4)

# Add all cells from part2 to part1
nb1.cells.extend(nb2.cells)

# Save the merged notebook
with open('Assignments/Assignment_1.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb1, f)