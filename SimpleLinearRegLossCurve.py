#x,y={'x1':[3.5,3.69,3.44,3.43,4.34,4.42,2.37]},[18,15,18,16,15,14,24]    # y-hat vector still empty
     # bias and weight initialized at 0, learning rate to be specified by user
x,y,Y={'x1':[1,2],'x2':[3,4],'x3':[5,6]},[8,9],[] # TEST DATA
def SimpleLinRegLossCurve(x,y,Epoch,LR):
    leny=len(y)
    Y=[]
    epoch=0
    import numpy as np
    bias=0
    weight=[]
    for i in x.keys():
        weight.append(0)
    for a in range(0,len(y)):       # for each 'dataset row'
        curvec=[]
        for b in x.keys():
            curvec.append(x[b][a])  # collect feature values for that row in a vector
        Y.append(int(sum(np.array(weight)*np.array(curvec))+bias)) # multiply that vector
        # by weight vector and add bias, i.e getting y-hat value, then append y-hat value to the
        # y-hat vector
    mseloss=round(sum((np.array(y)-np.array(Y))**2)/len(y), 2)        # calculate initial loss
    print(f'first y-hat vector = {Y} and first loss value = {mseloss}')
    def UpWeight(Y,y,x,n):      # n = numerical size of data
        wslopesD={}     # dictionary to hold features and their slopes
        for i in x.keys():
            wslopesD[i]=[]      # each feature has a key value of an empty list
            for j in range(0,len(y)):
                wslopesD[i].append((Y[j]-y[j])*(2*x[i][j])) # solve for upper half of weight slope
                # formula
        wslopes=[]      # list to hold actual slope values
        for i in wslopesD.values():
            wss=sum(i)/len(y)       # slope values calculated by dividing by 'n'
            wslopes.append(wss)     # slopes contained in the list
        nweights=[]     # list to hold new weight values
        for i in range(0,len(x)):
            nweights.append(round(weight[i]-(wslopes[i]*LR),2)) # new weights calculated using 
            # new weight formula
        return nweights
    def UpBias(Y,y,n):
        BSL=[]
        for i in range (0,len(y)):
            BSL.append((Y[i]-y[i])*2)
        bslope=sum(BSL)/len(y)
        nbias=bias-(LR*bslope)
        return round(nbias,2)                # first bias update
    losses=[]           # for graph
    epochs=[]           # for graph
    while epoch<Epoch:
        weight=UpWeight(Y,y,x,len(x))
        bias=UpBias(Y,y,len(x))
        print(f'b={bias}, w={weight}')
        Y=[]
        for i in range(0,leny):
            fts=[]
            for j in x.keys():
                fts.append(x[j][i])
            in_y=float(sum(np.array(weight)*np.array(fts))+bias)
            Y.append(round((in_y),2))          # continue updating y-hat values to calculate loss
        mseloss=round(sum((np.array(y)-np.array(Y))**2)/len(y),2)           # calculate and update loss
        print(f'current y-hat = {Y}, and current loss value = {mseloss}')
        losses.append(mseloss)
        epochs.append(epoch)
        epoch+=1
        
    import matplotlib.pyplot as plt
    plt.plot(epochs, losses, linestyle='-', color='b')
    plt.show()

SimpleLinRegLossCurve(x,y,8,0.005)
