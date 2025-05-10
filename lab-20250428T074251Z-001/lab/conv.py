import numpy as np
def zConv(m,K):
#input assumed to be numpy arrays Kr<=mrow, Kc<=mcol, Kernal odd
#edges wrap Top/Bottom, Left/Right
#Zero Pad m by kr,kc if no wrap desired
  mc=m*0
  Kr,Kc= K.shape
  kr=Kr//2 #kernel center
  kc=Kc//2
  for dr in range(-kr,kr+1):
    mr=np.roll(m,dr,axis=0)
    for dc in range(-kc,kc+1):
      mrc=np.roll(mr,dc,axis=1)
      mc=mc+K[dr+kr,dc+kc]*mrc
  return mc
arr1 = np.array([[1,2,	5,	2],
                [2,	3,	4,	9],
                [7,	8,	0,	1],
                [4,	1,	3,	4]])
arr2 = np.array([[1,	2,	-3],
                 [2,	-7,	5]
                ])

arr_result = zConv(arr1, arr2)

print(arr_result)
