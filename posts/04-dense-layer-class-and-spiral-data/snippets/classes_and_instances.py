"""Post 04, section 3: a class, two instances, and what self refers to.

Run from the series root:
    python posts/04-dense-layer-class-and-spiral-data/snippets/classes_and_instances.py
"""


class Dog:
    def __init__(self, name):
        self.name = name

a = Dog("Buddy")    # __init__ runs with self = a, name = "Buddy"
b = Dog("Lucy")     # __init__ runs again with self = b, name = "Lucy"
print(a.name, b.name)   # "Buddy Lucy"

a.name = "Rex"          # change one instance
print(a.name, b.name)   # "Rex Lucy": b is untouched
