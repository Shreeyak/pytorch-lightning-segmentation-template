from collections import deque

a = deque(maxlen=3)
a.append(1)
a.append(2)
a.append(3)
print(a)
a.append(4)
print(a)


z = [x for x in a]
print(z)
a.append(5)
print(a)
print(z)
print(len(a))
a.pop()
print(len(a))
