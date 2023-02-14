from numpy import array, tensordot

biais = 999
T = array([
  [[1,2,3,biais],    [4,5,6,biais],    [7,8,9,biais]],
  [[11,12,13,biais], [14,15,16,biais], [17,18,19,biais]],
  [[21,22,23,biais], [24,25,26,biais], [27,28,29,biais]],
    [[30,31,32,biais],[33,34,35,biais],[36,37,38,biais]]
  ])

print(T.shape)
print(T)

A = array([1,2])
B = array([3,4])

C = tensordot(A,B, axes=0)
print(C) #[3,4,6,8]