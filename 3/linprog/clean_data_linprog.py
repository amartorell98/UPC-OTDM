#!/usr/bin/env python3
"""
Auxiliary script to clean data into AMPL readable .dat files for the problem at hand.
"""
import sys
import numpy as np
import pandas as pd
import json

def arrange_dat(dat_file, k, line_split=None):
    with open(dat_file, "r") as f:
        m = 1
        xs = None
        for line in f:
            point = line.split(line_split)
            point = list(map(lambda s: s.strip(), point))
            x = [float(x) for x in point]

            if xs is None:
                xs = x
            else:
                xs = np.vstack([xs, x])

            m += 1

    m = xs.shape[0]
    d = xs.shape[1]
    print(f"param m:={m};")
    print(f"param k:={k};")
    xranges = ' '.join(map(str, list(range(1, d+1))))
    print(f"param d: {xranges}:=")
    for i in range(1, m+1):
        if i < m:
            print(i, list_to_string(xs[i-1]))
        else:
            print(i, list_to_string(xs[i-1]), ";")

def list_to_string(list):
    string = ""
    for elem in list:
        string += " " + str(elem)
    return string

if __name__ == "__main__":
    dat_file = sys.argv[1]
    k = sys.argv[2]
    if sys.argv[3] == "None":
        line_split = None
    else:
        line_split = sys.argv[3]
    
    arrange_dat(dat_file, k, line_split)
