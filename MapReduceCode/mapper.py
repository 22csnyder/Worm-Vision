#!/usr/bin/env python

import sys
import os

sys.path.append('/home/csnyder/default')


longdocname=os.getenv('mapreduce_map_input_file')


docname=longdocname[57:]

# input comes from STDIN (standard input)
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split()
    # increase counters
    for word in words:
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is 1
        print '%s.%s\t%s' % (docname,word, 1)
