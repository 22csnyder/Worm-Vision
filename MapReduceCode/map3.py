#!/usr/bin/env python

import sys
import os
import string

sys.path.append('/home/csnyder/default')



# input comes from STDIN (standard input)
for line in sys.stdin:
	line=line.strip()
	words=line.split()
        print '%s\t%s %s' % (words[0],words[1],words[2])
