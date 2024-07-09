#!/usr/bin/env python

import sys
import os
import string

sys.path.append('/home/csnyder/default')


# input comes from STDIN (standard input)
for line in sys.stdin:
	idx0=line.find('.')#this part needs to be changed if not webspam
	idx=line.find('.',idx0+1)#finds the next '.' for webspam because documents end in .txt
	idxtab=line.find('\t')
	document=line[:idx]
	term=line[idx+1:idxtab]
	TinD=line[idxtab+1:]
        print '%s\t%s %s' % (document,term,TinD)
