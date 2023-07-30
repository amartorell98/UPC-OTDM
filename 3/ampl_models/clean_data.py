#!/usr/bin/env python3
"""
Auxiliary script to clean data generated with the generator provided for the assignmentas well as other data structes into AMPL readable .dat files.
"""
import sys
import numpy as np
import pandas as pd
import json

def arrange_dat(dat_file, nu, line_split=None, y_map=None, train_id=None, print_train=True):
    with open(dat_file, "r") as f:
        m = 1
        err = 0
        xs = None
        for line in f:
            if print_train and m > train_id:
                break
            if not print_train and m <= train_id:
                m += 1
                continue
            point = line.split(line_split)
            point = list(map(lambda s: s.strip(), point))
            x = [float(x) for x in point[:-1]]
            y = point[-1]

            if '*' in y:
                y = float(y[:-1])
                err += 1
            else:
                if y_map:
                    y = float(y_map[y])
                else:
                    y = float(y)

            if xs is None:
                xs = x
                ys = y
            else:
                xs = np.vstack([xs, x])
                ys = np.vstack([ys, y])

            m += 1

    m = xs.shape[0]
    n = xs.shape[1]
    print(f"param nu:={nu};")
    print(f"param m:={m};")
    print(f"param n:={n};")
    xranges = ' '.join(map(str, list(range(1, n+1))))
    print(f"param x: {xranges}:=")
    for i in range(1, m+1):
        if i < m:
            print(i, list_to_string(xs[i-1]))
        else:
            print(i, list_to_string(xs[i-1]), ";")
    print("param y:=")
    for i in range(1, m+1):
        if i < m:
            print(i, list_to_string(ys[i-1]))
        else:
            print(i, list_to_string(ys[i-1]), ";")

def list_to_string(list):
    string = ""
    for elem in list:
        string += " " + str(elem)
    return string

if __name__ == "__main__":
    dat_file = sys.argv[1]
    nu = sys.argv[2]
    if sys.argv[3] == "None":
        line_split = None
    else:
        line_split = sys.argv[3]
    try:
        y_map = json.loads(sys.argv[4])
    except:
        y_map = None
    train_id = int(sys.argv[5])
    print_train = json.loads(sys.argv[6].lower())
    
    arrange_dat(dat_file, nu, line_split, y_map, train_id, print_train)
