import numpy as np
from pomdp_fetch import fetch

F = fetch([1,2,3],{1:{'bowl','wooden','brown'}, 2:{'fork','metal','silver'}, 3:{'spoon','metal','silver'}})


print('transition tests:')
t1 = F.transition([(1,None),(3,None)], [(1,None), (1,1), (3,1), (1,3), (3, None)], ('pick', 1))
a1 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
print(np.array_equal(t1, a1))

t2 = F.transition([(1,None),(3,None)], [(1,None), (1,1), (3,1), (1,3), (3, None)], ('point', 1))
a2 = np.array([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]])
print(np.array_equal(t2, a2))

print()
print('reward tests:')
t3 = F.reward([(1,None),(1,3),(2,None),(3,3)], ('pick',1))
a3 = np.array([10.0,10.0,-12.5,-12.5])
print(np.array_equal(t3,a3))

t4 = F.reward([(1,None),(1,3),(2,None),(3,3)], ('pick',2))
a4 = np.array([-12.5,-12.5,10.0,-12.5])
print(np.array_equal(t4,a4))

t5 = F.reward([(1,None),(1,3),(2,None),(3,3)], ('point',2))
a5 = np.array([-6.0,-6.0,-6.0,-6.0])
print(np.array_equal(t5,a5))

t6 = F.reward([(1,None),(1,3),(2,None),(3,3)], ('wait',2))
a6 = np.array([-1.0,-1.0,-1.0,-1.0])
print(np.array_equal(t6,a6))