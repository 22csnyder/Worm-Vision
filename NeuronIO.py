# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 00:52:35 2015

@author: csnyder
"""



#from traits.api import HasTraits



#class TemplateIO:
#    def __init__(self,save_dict):
#        self.dict=save_dict
        
        
    
####BLEH
    
import csv
class NeuronIO():
    
    folder=None
    
    def save_csv(self,data_labels,data,name):
        
    
    def save_intensity_as_csv(labels,data,name='intensity.csv'):
            f=open(folder+'/'+name, 'wb')
            c = csv.writer(f)
            for label,datum in zip(labels,data):
                c.writerow( [label]  +  list(datum) )
            f.close()
            
            
    def save_position_as_csv(labels,data,folder,name='neuron_position.csv'):
            f=open(folder+'/'+name, 'wb')
            c = csv.writer(f)
            for label,datum in zip(labels,data):
                c.writerow( [label]  +  list(datum) )
            f.close()
        



    
#class Template:
#    _template = """<ul>\n%(list|li)s\n</ul>"""
#    def __init__(self, dict):
#        self.dict = dict
#    def __str__(self):
#        return self._template % self
#    def __getitem__(self, key):
#        l = key.split("|")
#        if len(l) == 1:
#            return self.dict[key]
#        else:
#            return getattr(self, l[1])(self.dict[l[0]])
#    def li(self, l):
#        return "\n".join(["\t<li>%s</li>" % x for x in l])
#print Template({"list": ["foo", "bar", "baz"]})    
#t=Template({"list": ["foo", "bar", "baz"]})    
#print t._template % t


#class Template:
#
#    def __init__(self, dict):
#        self.dict = dict
#
#    def __str__(self):
#        return "Hello %(name)s. Hello %(name|upper)s!" % self
#
#    def __getitem__(self, key):
#        l = key.split("|")
#        if len(l) == 1:
#            return self.dict[key]
#        else:
#            return getattr(self, l[1])(self.dict[l[0]])
#
#    def upper(self, s):
#        return s.upper()
#
#
#print Template({"name": "Guido"})
#
#t=Template({"name": "Guido"})