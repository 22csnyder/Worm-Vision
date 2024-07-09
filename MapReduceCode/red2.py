#!/usr/bin/env python

from operator import itemgetter
import sys
#import array
#import math

term_count = 0#counts how many terms are in a given doc
term_count_keeper=[]
idxdoc=[]
docnumber=-1

d=[]
t=[]
TofD=[]

currentdoc=-1

# input comes from STDIN
for line in sys.stdin:
	line=line.strip()	
	if len(line)==0:continue
	idxtab=line.find('\t')
	idxspace=line.find(' ')
	doc=line[:idxtab]
	d.append(doc)
	t.append(line[idxtab+1:idxspace])
	TofD.append(line[idxspace:])
	
	
	
	if currentdoc!=doc:
		currentdoc=doc
		docnumber+=1
		term_count_keeper.append(0)
		
	term_count_keeper[docnumber]+=1
	idxdoc.append(docnumber)
	
	#print '%s %s %s' % (idxtab,idxspace,TofD[term_count-1])
#compute and output
#term document tf(t,d)





for i in xrange(len(t)):
	#print i
	#print len(idxdoc)
	#print idxdoc[i]	
	#print len(term_count_keeper)
	x=float(TofD[i])
	y=float(term_count_keeper[int(idxdoc[i])])
	tf=x/y    
	print '%s %s %s' % (t[i],d[i],tf)
