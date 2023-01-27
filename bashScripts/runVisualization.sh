#!/bin/bash

xNPZ='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Sub_300-400_XnuX.npz'
pref='200-300'

znpz="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Experiments/Sub_${pref}__0__optimalZnu_ZAllwC1.2_sig0.05_Nmax5000.0_Npart1000.0.npz"

python3 -c "import visualizeResample as vs; vs.writeParticleVTK('$znpz'); quit()"

znpz="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Experiments/Sub_${pref}__1__optimalZnu_ZAllwC1.2_sig0.05_Nmax5000.0_Npart1000.0.npz"

python3 -c "import visualizeResample as vs; vs.writeParticleVTK('$znpz'); quit()"

znpz="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Experiments/Sub_${pref}__2__optimalZnu_ZAllwC1.2_sig0.05_Nmax5000.0_Npart1000.0.npz"

python3 -c "import visualizeResample as vs; vs.writeParticleVTK('$znpz'); quit()"

znpz="/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Experiments/Sub_${pref}__3__optimalZnu_ZAllwC1.2_sig0.05_Nmax5000.0_Npart1000.0.npz"

python3 -c "import visualizeResample as vs; vs.writeParticleVTK('$znpz'); quit()"

