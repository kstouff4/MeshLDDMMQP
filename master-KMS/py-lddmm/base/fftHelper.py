# -*- coding: utf-8 *-*

import numpy
#import fftw3

def applyKernel(right, dims, num_nodes, Kv, el_vol, scale=True):
    """
    Apply the V kernel to momentum, with numpy fft, used for multi-threaded
    processing.
    """
    krho = numpy.zeros((num_nodes, len(dims)))
    for j in range(len(dims)):
      rr = right[:,j].copy().astype(complex)
      fr = numpy.reshape(rr, dims)
      fr = numpy.fft.fftshift(fr)
      fr = numpy.fft.fftn(fr)
      Kv_shift = numpy.fft.fftshift(Kv)
      fr = fr * Kv_shift
      out = numpy.fft.ifftn(fr)
      if scale:
          out *= 1./el_vol
      out = numpy.fft.fftshift(out)
      krho[:,j] = numpy.reshape(out.real, (num_nodes))
    return krho.real

# initialize fftw information
#        self.fft_thread_count = 8
#        self.in_vec = numpy.zeros(self.rg.dims, dtype=complex)
#        self.fft_vec = numpy.zeros(self.rg.dims, dtype=complex)
#        self.out_vec = numpy.zeros(self.rg.dims, dtype=complex)
#        self.wfor = fftw3.Plan(self.in_vec, self.fft_vec, \
#                                direction='forward', flags=['measure'], \
#                                )
#        self.wback = fftw3.Plan(self.fft_vec, self.out_vec, \
#                                direction='backward', flags=['measure'], \
#                                )

#def apply_kernel_V_for_async_w(right, dims, num_nodes, Kv, el_vol):
#    """
#    Apply the V kernel to momentum, with fftw, used for multi-threaded
#    processing.
#    """
#    in_vec = numpy.zeros(dims, dtype=complex)
#    fft_vec = numpy.zeros(dims, dtype=complex)
#    out_vec = numpy.zeros(dims, dtype=complex)
#
#    wfor = fftw3.Plan(in_vec, fft_vec, \
#                            direction='forward', flags=['estimate'], \
#                            )
#    wback = fftw3.Plan(fft_vec, out_vec, \
#                            direction='backward', flags=['estimate'], \
#                            )
#    krho = numpy.zeros((num_nodes, 3))
#    for j in range(3):
#        rr = right[:,j].copy().astype(complex)
#        fr = numpy.reshape(rr, dims)
#        fr = numpy.fft.fftshift(fr)
#        in_vec[...] = fr[...]
#        wfor.execute()
#        fr[...] = fft_vec[...]
#        Kv = numpy.fft.fftshift(Kv)
#        fr = fr * Kv
#        fft_vec[...] = fr[...]
#        wback.execute()
#        out = out_vec[...] / in_vec.size
#        out *= 1./el_vol
#        out = numpy.fft.fftshift(out)
#        krho[:,j] = out.real.ravel()
#    return krho


#    def apply_kernel_V(self, right):
#        rg, N, T = self.get_sim_data()
#        krho = numpy.zeros((rg.num_nodes, 3))
#        for j in range(self.dim):
#            rr = right[:,j].copy().astype(complex)
#            fr = numpy.reshape(rr, rg.dims)
#            fr = numpy.fft.fftshift(fr)
#            fr *= rg.element_volumes[0]
#            self.in_vec[...] = fr[...]
#            self.wfor.execute()
#            fr[...] = self.fft_vec[...]
#            Kv = self.get_kernelv()
#            Kv = numpy.fft.fftshift(Kv)
#            fr = fr * Kv
#            self.fft_vec[...] = fr[...]
#            self.wback.execute()
#            out = self.out_vec[...] / self.in_vec.size
#            out *= 1/rg.element_volumes[0]
#            out = numpy.fft.fftshift(out)
#            krho[:,j] = out.real.ravel()
#        return krho
