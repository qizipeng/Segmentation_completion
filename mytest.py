import numpy as np






if __name__ =="__main__":


    a = [[[1,1],[2,2],[3,3]],
         [[4,4],[5,5],[6,6]]]

    mask = [[1,0,1],
            [1,0,1]]

    a = np.array(a)
    mask = np.array(mask)

    a =a.reshape(6,-1)
    mask = mask.reshape(6)


    valid = np.where(mask == 1,True, False)
    print(valid)
    print(a[valid==True,:])


