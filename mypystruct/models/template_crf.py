from .graph_crf import GraphCRF
from ..utils import make_grid_edges#, edge_list_to_features
import numpy as np
from ..utils import expand_sym#, compress_sym
from ..inference import inference_dispatch
from .utils import loss_augment_unaries

class TemplateCRF(GraphCRF):

    def __init__(self, n_states=None, n_features=None, inference_method=None,
                 neighborhood=4,template=None):
        self.v=template
#        self._v=self.v.reshape(-1,1)
        self._v=self.v.ravel()
        self.neighborhood = neighborhood
        GraphCRF.__init__(self, n_states=n_states, n_features=n_features,
                          inference_method=inference_method)
#        self.initialize()
#        self.size_joint_feature+=1#gebroken   
    def initialize(self, X, Y):
        # Works for both GridCRF and GraphCRF, but not ChainCRF.
        # funny that ^^
        n_features = X[0][0].shape[1]
        if self.n_features is None:
            self.n_features = n_features
        elif self.n_features != n_features:
            raise ValueError("Expected %d features, got %d"
                             % (self.n_features, n_features))

        n_states = len(np.unique(np.hstack([y.ravel() for y in Y])))
        if self.n_states is None:
            self.n_states = n_states
        elif self.n_states != n_states:
            raise ValueError("Expected %d states, got %d"
                             % (self.n_states, n_states))

        self._set_size_joint_feature()
        self._set_class_weight()
        self.offset_pairwise=self.n_features*self.n_states
        self.offset_template=self.offset_pairwise + self.n_states*(self.n_states+1)/2

    def _set_size_joint_feature(self):
        # try to set the size of joint_feature if possible
        if self.n_features is not None and self.n_states is not None:
            if self.directed:
                self.size_joint_feature = (self.n_states * self.n_features +
                                           self.n_states ** 2)
            else:
                self.size_joint_feature = (
                    self.n_states * self.n_features
                    + self.n_states * (self.n_states + 1) / 2)
                    
            self.size_joint_feature+=1#For n_templates


    def _get_pairwise_potentials(self, x, w):  
        """Computes pairwise potentials for x and w.
##changed to reflect that pairwise potentials don't extend to the end of w
        Parameters
        ----------
        x : tuple
            Instance Representation.

        w : ndarray, shape=(size_joint_feature,)
            Weight vector for CRF instance.

        Returns
        -------
        pairwise : ndarray, shape=(n_states, n_states)
            Pairwise weights.
        """
        self._check_size_w(w)
        self._check_size_x(x)
        pw = w[self.offset_pairwise:self.offset_template]
        if self.directed:
            print 'Error: template_crf:offsets in w not implemented for directed graphs'
            return pw.reshape(self.n_states, self.n_states)
        return expand_sym(pw)
        
    def _get_template_potentials(self, x,w):
        self._check_size_w(w)
        self._check_size_x(x)
        return w[self.offset_template:]
        
    def _get_edges(self, x):
        return make_grid_edges(x, neighborhood=self.neighborhood)

    def _get_features(self, x):
        return x.reshape(-1, self.n_features)

    def _reshape_y(self, y, shape_x, return_energy):
        if return_energy:
            y, energy = y

        if isinstance(y, tuple):
            y = (y[0].reshape(shape_x[0], shape_x[1], y[0].shape[1]), y[1])
        else:
            y = y.reshape(shape_x[:-1])

        if return_energy:
            return y, energy
        return y

    def inference(self, x, w, relaxed=False, return_energy=False):
#        print 'inference is called'

        self._check_size_w(w)
        self.inference_calls += 1
        unary_potentials = self._get_unary_potentials(x, w)
#        print 'unary_potentials in inference'
#        print unary_potentials
        t=self._get_template_potentials(x,w)
#        template_potentials=np.hstack([np.zeros(self._v.shape),-self.v,self.v])*t
#        unary_potentials+=template_potentials
#        print 'unary_potentials shape',unary_potentials.shape
#        print '_v shape',self._v.shape
#        print 't shape',t.shape
        
        unary_potentials[:,1]-=self._v*t
        unary_potentials[:,2]+=self._v*t
        
#        template=template_potentials*self.v#should just be scalar right now
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        edges = self._get_edges(x)
#        print 'feed into inference_dispatch unaries',unary_potentials.shape
        y=inference_dispatch(unary_potentials=unary_potentials,
                                  pairwise_potentials=pairwise_potentials,
#                                  template_potentials=template,
                                  edges=edges,
                                  inference_method=self.inference_method,
                                  relaxed=relaxed,
                                  return_energy=return_energy)
        return self._reshape_y(y, x.shape, return_energy)                         

    def loss_augmented_inference(self, x, y, w, relaxed=False,return_energy=False):
        if self.inference_method not in ['ogm','qpbo']:
            print 'inference method not supported yet for templateCRF'
#        print 'loss_aug inference is called'
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self._get_unary_potentials(x, w)
#        print 'length unary_potentials is',len(unary_potentials)
#        print 'unary_potentials shape',unary_potentials.shape
#        print '_v shape',self._v.shape
        pairwise_potentials = self._get_pairwise_potentials(x, w)
        template_potentials=self._get_template_potentials(x,w)
        template_unaries=self._v*template_potentials#should just be scalar right now
        unary_potentials[:,1]-=template_unaries
        unary_potentials[:,2]+=template_unaries
        edges = self._get_edges(x)
        loss_augment_unaries(unary_potentials, np.asarray(y.ravel()), self.class_weight)
#        print 'shape loss_augment_unaries',unary_potentials.shape
        y = inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)
        return self._reshape_y(y,x.shape,return_energy)
        


    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y_hat is the result of linear programming
        'continuous loss is called inside template_crf'
        return GraphCRF.continuous_loss(
            self, y.ravel(), y_hat.reshape(-1, y_hat.shape[-1]))
    
#    def _set_size_joint_feature(self):
#        GraphCRF._set_size_joint_feature(self)
#        self.size_joint_feature+=1#gebroken


    def joint_feature(self,x,y):
        if isinstance(y,tuple):
            print 'lp inference not implemented for template_crf'
            a,b=y
            print a.shape
            print b.shape
        a=np.dot(self.v.ravel(), (y==1).ravel() )
        b=np.dot(self.v.ravel(), (y==2).ravel() )
        return np.hstack([GraphCRF.joint_feature(self,x,y),b-a])#I don't know about "self" here
