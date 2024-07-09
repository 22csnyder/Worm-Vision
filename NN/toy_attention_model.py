
import numpy as np
import theano
import theano.tensor as T

from skimage import io

###########TOGGLE THESE ON AND OFF#########
#theano.config.compute_test_value='warn'#Use tag.test_value for debug
theano.config.compute_test_value='off'#Running
###########################################

Pa=theano.Param


def floatX(X):
    return np.asarray(X,dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape)*0.01))

H1=np.zeros((4,4));H2=np.zeros((4,4));H3=np.zeros((4,4));H4=np.zeros((4,4))
H1[:2,:2]=1;H2[:2,2:]=1;H3[2:,:2]=1;H4[2:,2:]=1
W=0.25*np.vstack([H1.ravel(),H2.ravel(),H3.ravel(),H4.ravel()])

A_default=W.astype(theano.config.floatX)

I=np.array([
    [1,2,1,2],
    [1,1,1,1],
    [1,2,2,2],
    [1,1,2,1]],dtype=np.float)
    
#I/=I.max()
I-=1
I=floatX(I)

Cost=np.array([
    [1,1,1,0],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]],dtype=np.float)



Qave=T.matrix('Qave')
Image=T.matrix('Image')
h1=T.dot(Qave,Image.ravel())
ave_quadrants = theano.function([Image,Pa(Qave,default=A_default)],h1)



#####Get Focii Coordinates Given Output######
#numpy precompute
delta=0.5#(half)stride
#center_min=delta
#center_max=delta
np_gap=floatX((np.indices((2,2)).reshape(2,4).transpose()-0.5)*delta)
##Make filters 
h=T.vector('h')
offsets=T.matrix('offsets')
#theano.config.compute_test_value='warn'
#h.tag.test_value=floatX(np.random.rand(2))
#offsets.tag.test_value=floatX(np_gap)
#center=2*h+1
center=4*h
#center=h#debug
b_center=center.dimshuffle('x',0)
focii=b_center+offsets
focii_coordinates = theano.function([ h, Pa(offsets,default=np_gap) ],focii , allow_input_downcast=True)
#focii_coordinates = theano.function([ h ],focii)


#theano.config.compute_test_value='warn'
#h_1=T.vector('h_1')
#h_1.tag.test_value=floatX(np.array([0.4,0.6]))
#a=focii_coordinates(h_1)
#####Get Focii Coordinates Given Output######



#########Get Filters Given Focus##########
###numpy##
#xi,yi=np.indices((4,4))
#u=np.array([1.0,1.0])##With sigma=0.4 we don't have any patches larger than 2x2
#sigma=0.4
#E=np.exp(-(  (u[0]-xi)**2 + (u[0]-yi)**2  )/(2*sigma**2))#sigma=1
#E/=np.sum(E)
#E[E<0.01]=0
#print E
np_indices=floatX(np.indices((4,4)).transpose(1,2,0))
np_sig=floatX(0.4)
##theano##
sigma=T.scalar('sigma')
Index=T.tensor3('Index')
u=T.matrix('u')
#theano.config.compute_test_value='warn'
#u.tag.test_value=floatX(np.array([[1.5,1.0],
#                                  [0.75,0.75],
#                                  [2.9,0.2]  ]))
#sigma.tag.test_value=floatX(np_sig)
#Index.tag.test_value=floatX(np_indices)
b_u=u.dimshuffle('x','x',0,1)#'x' indicates additional dimensions that should be broadcast#automatically makes broadcastable (True,True,False)
b_Index=Index.dimshuffle(0,1,'x',2)
E=T.exp(-   T.sum( (b_u-b_Index)**2,axis=3) / (2*sigma**2)   )
E/=T.sum(E,axis=(0,1),keepdims=True)#slick
filters = theano.function([ u,Pa(Index,default=np_indices),Pa(sigma,default=np_sig)], E )
#########Get Filters Given Focus##########





#    cx=2.0*gx+1.0
#    cy=2.0*gy+1.0
#    center=np.array([cx,cy]) #between 1 and 3
#    focii=center+offsets
#    X=floatX(X)
#    return X

w_0=init_weights((4,2))
#w_1=init_weights((4,2))
#w_3=init_weights((4,4))
Y=T.matrix('Y')
X=T.matrix('X')
#theano.config.compute_test_value='warn'#"raise" is also an option
X.tag.test_value=floatX(I)
f0=ave_quadrants(X)
##f0=ave_quadrants(I)#debug
#h_1=T.nnet.sigmoid( T.dot(f0,w_0) )
##c1=focii_coordinates(h_1.tag.test_value)#debug
#c1=focii_coordinates(h_1)#debug
##X=I
#X.tag.test_value=I
#f1=filters(c1)#x,y,point(4,4,4)


#xr=X.ravel()
#b_X=xr.dimshuffle('x',0)
#b_X=_X.dimshuffle('x',0)


#debug
#F1=T.tensor3('F1')
#F1.tag.test_value=f1
#r_F1=F1.reshape((1,16,4))[0]
#features= T.dot( r_F1.transpose(), X.ravel() )
#h_2=T.nnet.sigmoid( T.dot(f0,w_0) )




#results,updates=theano.scan( fn=lambda focus: rbf(focus[0],focus[1]),
#                            outputs_info=None,
#                            sequences=[focii],
#                            non_sequences=None )



#def sgd(cost,params,lr=0.05):
#    grads=T.grad(cost=cost,wrt=params)
#    updates=[]
#    for p,g in zip(params,grads):
#        updates.append([p,p-g*lr])
#    return updates
#

#def model(I,w_1,w2):
#    X1=ave_quadrants(I)
#    h1=T.nnet.sigmoid( T.dot(X1,w_1) )
#    gx,gy=h1
#    X2=glimpse22(gx,gy)
#    y=T.nnet.sigmoid( T.dot(X2,w_2) )
#    return y

#def model(X,w_1):





#
#y=model(X,w_1,w_2)
#
#cost= - T.dot( Y,y )
#
#params=[w_1,w_2]
#updates=sgd(cost,params)




#train= theano.function( inputs=[X,Y],outputs=cost,updates=updates,allow_input_downcast=True)
#train= theano.function( inputs=[X],outputs=cost,updates=updates,allow_input_downcast=True)
#predict= theano.function(inputs=[X],outputs=y,allow_input_downcast=True)











#for i in range(200):
#    cost=train( I )
    




#def l1(X,w_1):
#    X1=np.array([[np.mean(X[:2,:2]),np.mean(X[:2,2:])],
#                [np.mean(X[2:,:2]),np.mean(X[2:,2:])]])
#    X1=floatX(X1.flatten())
#    h1=T.nnet.sigmoid( T.dot(X1,w_1) )
#    return h1
#
#def glimpse1(h1):
#    gx,gy=h*I.shape
#    
#
#
#def l2(X,focus)

    
    
#offsets=np.indices((2,2))*2-1
#focus=2*np.array([gx,gy])+1#between 1 and 3
#
#loci=offsets+focus
#for p in loci:
#cil=np.ceil(p)
#flr=np.floor(p)
#pix_surr=np.array([ [flr[0],cil[0]],[flr[1],cil[0]],[flr[0],cil[1]],[flr[1],cil[1]]] )
#
#wts=[ np.exp( -np.linalg.norm(q-p)**2 ) for q in pix_surr ]
#vals=[I[q] for q in pix_surr]  