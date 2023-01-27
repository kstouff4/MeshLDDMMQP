#!/bin/bash

s1r1='/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/cell_S1R1_mesh100.vtk'
s1r2='/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/cell_S1R2_mesh100.vtk'
s2r1='/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/cell_S2R1_mesh100.vtk'
s2r2='/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/cell_S2R2_mesh100.vtk'

python3 -c "import meshAlter as ma; ma.collapseLabelsProb('$s1r1',[[0,1,2,3,4,20,21,23,24,25,26,27,28,29,30,31],[5,6,10,11,12,13,14,15,16,17,18],[7,8,9,32]]); ma.collapseLabelsProb('$s1r2',[[0,1,2,3,4,20,21,23,24,25,26,27,28,29,30,31],[5,6,10,11,12,13,14,15,16,17,18],[7,8,9,32]]); ma.collapseLabelsProb('$s2r1',[[0,1,2,3,4,20,21,23,24,25,26,27,28,29,30,31],[5,6,10,11,12,13,14,15,16,17,18],[7,8,9,32]]); ma.collapseLabelsProb('$s2r2',[[0,1,2,3,4,20,21,23,24,25,26,27,28,29,30,31],[5,6,10,11,12,13,14,15,16,17,18],[7,8,9,32]]); quit()"