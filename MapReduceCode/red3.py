#!/usr/bin/env python

from operator import itemgetter
import sys
import math

doc_count_keeper = []
idxterm=[]
term_number=-1

d=[]
t=[]
tf=[]
current_term=-1

####HARDCODE#####
sizeD=300002
########


# input comes from STDIN
for line in sys.stdin:
	line=line.strip()	
	if len(line)==0:continue
	idxtab=line.find('\t')
	term=line[:idxtab]
	t.append(term)
	line2=line[idxtab+1:]
	words=line2.split()	
	d.append(words[0])
	tf.append(float(words[1]))

	if term!=current_term:
		current_term=term
		term_number+=1
		doc_count_keeper.append(0)
	
	doc_count_keeper[term_number]+=1
	idxterm.append(term_number)
	


for i in xrange(len(t)):
	#print i
	idf = math.log10(float(sizeD)/float(doc_count_keeper[idxterm[i]]))    
	print '%s %s %f' % (t[i],d[i],tf[i]*idf)
	#print '%s %s' % (t[i],d[i])



