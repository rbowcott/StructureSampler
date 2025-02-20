import torch as T

a = T.ones((5, 6, 3))

print(a.sum((1,2)))

